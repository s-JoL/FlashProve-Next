# SPDX-License-Identifier: Apache-2.0
"""Spool directory shared by the online server and the asynchronous prover daemon (provider side only).

    SPOOL/pending/<id>.job     written by the server (atomic rename), one pickled proof job per reply
    SPOOL/working/<owner>.lock  one per live prover daemon, held with fcntl.flock(LOCK_EX) for the daemon's lifetime
    SPOOL/working/<owner>/<id>.job  claimed by that daemon (atomic rename from pending/; several daemons may share
                               one spool: the rename succeeds for exactly one of them)
    SPOOL/ready/<id>.proof     proof bytes, present ONLY after the daemon's CPU self-check printed VERIFIED
    SPOOL/ready/<id>.json      {"sha256": ..., "verify": "VERIFIED", timings}
    SPOOL/failed/<id>.json     {"reason": ...}  (proof never released; a forced/bad proof is kept as failed/<id>.proof
                               for the operator's audit only)
    SPOOL/port_pages/          prefix-cache port pages of the server (vllm_capture_zk.PrefixPortCache)

All directories are created with mode 0700 and files with umask 077: a job holds the secret parts of the trace
(system prompt, hidden CoT, every layer boundary).  The server never waits for the daemon; the daemon never talks
to vLLM.  Proof ids are 32 lowercase hex characters (checked before any path is formed).

Several daemons on one spool.  Each daemon registers a unique owner name (``attach_worker``): it creates
and flocks working/<owner>.lock *before* creating working/<owner>/, and keeps the lock until it exits (the kernel
drops it if the process dies, however it dies).  ``requeue_stale`` moves a job back to pending/ only if its owner's
lock can be taken now (owner dead) or no longer exists; jobs of live daemons are never touched, so a daemon that starts
while another is proving cannot steal its job.  A requeuer unlinks a dead owner's lock while holding it; a daemon that
lost that race (it flocked an inode that was unlinked meanwhile) notices via fstat/stat and retries with a new name.
Jobs found directly in working/ (no owner directory) have no owner and are requeued.
"""
import fcntl
import hashlib
import json
import os
import pickle
import re
import secrets
import shutil
import time

STATES = ("pending", "working", "ready", "failed")
_ID = re.compile(r"^[0-9a-f]{32}$")
_OWNER = re.compile(r"^w[0-9]+-[0-9a-f]{8}$")


def valid_id(pid):
    return isinstance(pid, str) and bool(_ID.match(pid))


class Spool:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.owner, self._lock_fd = None, None
        old = os.umask(0o077)
        try:
            os.makedirs(self.root, mode=0o700, exist_ok=True)
            for s in STATES:
                os.makedirs(os.path.join(self.root, s), mode=0o700, exist_ok=True)
        finally:
            os.umask(old)
        os.chmod(self.root, 0o700)

    def path(self, state, pid, ext):
        assert state in STATES and valid_id(pid)
        return os.path.join(self.root, state, pid + ext)

    def _atomic_write(self, path, data):
        tmp = path + ".tmp.%d" % os.getpid()
        old = os.umask(0o077)
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush(); os.fsync(f.fileno())
        finally:
            os.umask(old)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- server side
    def submit(self, pid, job):
        self._atomic_write(self.path("pending", pid, ".job"), pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL))

    def status(self, pid):
        """-> (state, info) with state in pending / working / ready / failed / unknown."""
        if not valid_id(pid):
            return "unknown", {}
        rj = self.path("ready", pid, ".json")
        if os.path.exists(rj) and os.path.exists(self.path("ready", pid, ".proof")):
            info = json.load(open(rj))
            return ("ready", info) if info.get("verify") == "VERIFIED" else ("failed", {"reason": "self-check record invalid"})
        if os.path.exists(self.path("failed", pid, ".json")):
            return "failed", json.load(open(self.path("failed", pid, ".json")))
        if self._working_job(pid) is not None:
            return "working", {}
        if os.path.exists(self.path("pending", pid, ".job")):
            return "pending", {}
        return "unknown", {}

    def ready_bytes(self, pid):
        """Proof bytes, only if the stored self-check record matches the file (sha256)."""
        state, info = self.status(pid)
        if state != "ready":
            return None
        blob = open(self.path("ready", pid, ".proof"), "rb").read()
        if hashlib.sha256(blob).hexdigest() != info.get("sha256"):
            return None
        return blob

    # ---------------------------------------------------------------- daemon side
    def _wdir(self):
        return os.path.join(self.root, "working")

    def _working_job(self, pid):
        """Path of pid's job in working/ (any owner, or the flat pre-5C layout), or None."""
        flat = self.path("working", pid, ".job")
        if os.path.exists(flat):
            return flat
        try:
            owners = os.listdir(self._wdir())
        except FileNotFoundError:
            return None
        for o in owners:
            if _OWNER.match(o):
                p = os.path.join(self._wdir(), o, pid + ".job")
                if os.path.exists(p):
                    return p
        return None

    def attach_worker(self):
        """Register this process as a prover daemon: unique owner name, its lock held until the process exits."""
        assert getattr(self, "owner", None) is None, "worker already attached"
        old = os.umask(0o077)
        try:
            while True:
                owner = "w%d-%s" % (os.getpid(), secrets.token_hex(4))
                lp = os.path.join(self._wdir(), owner + ".lock")
                fd = os.open(lp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)        # nobody else waits on a fresh unique name except a requeuer
                try:
                    if os.fstat(fd).st_ino == os.stat(lp).st_ino:
                        break                          # still linked: the lock is ours
                except FileNotFoundError:
                    pass
                os.close(fd)                           # a requeuer unlinked it between O_CREAT and flock: retry
            os.write(fd, ("%d\n" % os.getpid()).encode())
            os.makedirs(os.path.join(self._wdir(), owner), mode=0o700)
        finally:
            os.umask(old)
        self.owner, self._lock_fd = owner, fd
        return owner

    def wpath(self, pid, ext):
        """Path inside this daemon's own working directory."""
        assert getattr(self, "owner", None) and valid_id(pid)
        return os.path.join(self._wdir(), self.owner, pid + ext)

    def claim_next(self):
        """Oldest pending job -> (pid, job) moved to working/<owner>/, or None.  Safe with several daemons."""
        d = os.path.join(self.root, "pending")
        names = []
        for n in os.listdir(d):
            if n.endswith(".job") and valid_id(n[:-4]):
                try:
                    names.append((os.path.getmtime(os.path.join(d, n)), n))
                except FileNotFoundError:           # claimed by another daemon meanwhile
                    pass
        for _, n in sorted(names):
            pid = n[:-4]
            try:
                os.replace(os.path.join(d, n), self.wpath(pid, ".job"))
            except FileNotFoundError:
                continue
            with open(self.wpath(pid, ".job"), "rb") as f:
                return pid, pickle.load(f)
        return None

    def requeue_stale(self):
        """Jobs left in working/ by daemons that are no longer alive go back to pending/ (live daemons' jobs stay).
        Returns the number of jobs requeued.  Called at daemon start and periodically while idle."""
        wd, moved = self._wdir(), 0
        for n in os.listdir(wd):
            full = os.path.join(wd, n)
            if n.endswith(".job") and valid_id(n[:-4]):           # pre-5C flat layout: no owner
                try:
                    os.replace(full, self.path("pending", n[:-4], ".job")); moved += 1
                except FileNotFoundError:
                    pass
                continue
            owner = n[:-5] if n.endswith(".lock") else n
            if not _OWNER.match(owner) or owner == getattr(self, "owner", None):
                continue
            lp, jd = os.path.join(wd, owner + ".lock"), os.path.join(wd, owner)
            if n.endswith(".lock") and os.path.isdir(jd):
                continue                                          # handled through its directory entry
            try:
                fd = os.open(lp, os.O_RDWR)
            except FileNotFoundError:
                fd = None                                         # no lock: owner gone (its lock is created first)
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue                                      # owner alive: never touch its jobs
            try:
                if os.path.isdir(jd):
                    for j in os.listdir(jd):
                        p = os.path.join(jd, j)
                        if j.endswith(".job") and valid_id(j[:-4]):
                            pid_ = j[:-4]
                            # If the dead daemon already published a valid ready pair (or failed record) right before
                            # unlinking its working job, discard the leftover working job instead of re-proving it.
                            st_now, _ = self.status(pid_)
                            if st_now in ("ready", "failed") and (st_now != "ready" or self.ready_bytes(pid_) is not None):
                                try:
                                    os.remove(p)
                                except FileNotFoundError:
                                    pass
                                continue
                            try:
                                os.replace(p, self.path("pending", pid_, ".job")); moved += 1
                            except FileNotFoundError:
                                pass
                        elif os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)  # a dead daemon's self-check scratch
                        else:
                            try:
                                os.remove(p)
                            except FileNotFoundError:
                                pass
                    try:
                        os.rmdir(jd)
                    except OSError:
                        pass
                if fd is not None:
                    try:
                        os.remove(lp)                             # unlinked while we hold it (see attach_worker)
                    except FileNotFoundError:
                        pass
            finally:
                if fd is not None:
                    os.close(fd)
        return moved

    def detach_worker(self):
        """Clean exit of a daemon: its (now empty) working directory and lock are removed."""
        if getattr(self, "owner", None) is None:
            return
        jd = os.path.join(self._wdir(), self.owner)
        left = [j for j in os.listdir(jd) if j.endswith(".job")] if os.path.isdir(jd) else []
        if not left:
            shutil.rmtree(jd, ignore_errors=True)
            try:
                os.remove(os.path.join(self._wdir(), self.owner + ".lock"))
            except FileNotFoundError:
                pass
        os.close(self._lock_fd)                                   # jobs still left are requeued by the next daemon
        self.owner = None

    def finish_ready(self, pid, blob, info):
        if self.ready_bytes(pid) is not None:                     # never overwrite an already published valid proof
            try:
                os.remove(self.wpath(pid, ".job"))
            except FileNotFoundError:
                pass
            return
        self._atomic_write(self.path("ready", pid, ".proof"), blob)
        info = dict(info, sha256=hashlib.sha256(blob).hexdigest(), verify="VERIFIED", finished=time.time())
        self._atomic_write(self.path("ready", pid, ".json"), json.dumps(info).encode())
        try:
            os.remove(self.wpath(pid, ".job"))
        except FileNotFoundError:
            pass

    def finish_failed_direct(self, pid, reason, info=None):
        """Server side: a reply whose trace could not be recorded (never queued); its proof is reported withheld."""
        rec = dict(info or {}, reason=reason, finished=time.time())
        self._atomic_write(self.path("failed", pid, ".json"), json.dumps(rec, default=str).encode())

    def finish_failed(self, pid, reason, blob=None, info=None):
        if blob is not None:
            self._atomic_write(self.path("failed", pid, ".proof"), blob)
        rec = dict(info or {}, reason=reason, finished=time.time())
        self._atomic_write(self.path("failed", pid, ".json"), json.dumps(rec, default=str).encode())
        try:
            os.remove(self.wpath(pid, ".job"))
        except FileNotFoundError:
            pass
