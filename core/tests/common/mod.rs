// SPDX-License-Identifier: Apache-2.0
//! Minimal JSON reader for the golden vectors (keeps fpn-core free of dev-dependencies).
#![allow(dead_code, clippy::needless_range_loop)]

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub enum J {
    Null,
    Bool(bool),
    Num(i64),
    Str(String),
    Arr(Vec<J>),
    Obj(BTreeMap<String, J>),
}

impl J {
    pub fn get(&self, k: &str) -> &J {
        match self {
            J::Obj(m) => m.get(k).unwrap_or_else(|| panic!("missing key {k}")),
            _ => panic!("not an object"),
        }
    }
    pub fn has(&self, k: &str) -> bool {
        matches!(self, J::Obj(m) if m.contains_key(k))
    }
    pub fn arr(&self) -> &Vec<J> {
        match self {
            J::Arr(a) => a,
            _ => panic!("not an array"),
        }
    }
    pub fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }
    pub fn num(&self) -> i64 {
        match self {
            J::Num(n) => *n,
            _ => panic!("not a number"),
        }
    }
    pub fn bool(&self) -> bool {
        match self {
            J::Bool(b) => *b,
            _ => panic!("not a bool"),
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, J::Null)
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.b[self.i], c, "expected {} at {}", c as char, self.i);
        self.i += 1;
    }
    fn value(&mut self) -> J {
        self.ws();
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.b[self.i] == b'}' {
                    self.i += 1;
                    return J::Obj(m);
                }
                loop {
                    self.ws();
                    let k = match self.value() {
                        J::Str(s) => s,
                        _ => panic!("object key"),
                    };
                    self.eat(b':');
                    let v = self.value();
                    m.insert(k, v);
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    } else {
                        self.eat(b'}');
                        return J::Obj(m);
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b[self.i] == b']' {
                    self.i += 1;
                    return J::Arr(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    } else {
                        self.eat(b']');
                        return J::Arr(a);
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut s = String::new();
                while self.b[self.i] != b'"' {
                    let c = self.b[self.i];
                    assert!(c != b'\\', "escapes are not used in the golden vectors");
                    s.push(c as char);
                    self.i += 1;
                }
                self.i += 1;
                J::Str(s)
            }
            b't' => {
                self.i += 4;
                J::Bool(true)
            }
            b'f' => {
                self.i += 5;
                J::Bool(false)
            }
            b'n' => {
                self.i += 4;
                J::Null
            }
            _ => {
                let st = self.i;
                while self.i < self.b.len()
                    && (self.b[self.i] == b'-' || self.b[self.i].is_ascii_digit())
                {
                    self.i += 1;
                }
                J::Num(
                    std::str::from_utf8(&self.b[st..self.i])
                        .unwrap()
                        .parse()
                        .unwrap(),
                )
            }
        }
    }
}

pub fn parse(s: &str) -> J {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    let v = p.value();
    p.ws();
    assert_eq!(p.i, p.b.len(), "trailing data");
    v
}

pub fn load(name: &str) -> J {
    let path = format!("{}/../specs/vectors/{}", env!("CARGO_MANIFEST_DIR"), name);
    parse(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
}

pub fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2));
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// "0x..." hex integer as little-endian u64 limbs.
pub fn limbs(s: &str) -> Vec<u64> {
    let h = s.strip_prefix("0x").expect("0x prefix");
    let mut out = Vec::new();
    let mut end = h.len();
    while end > 0 {
        let st = end.saturating_sub(16);
        out.push(u64::from_str_radix(&h[st..end], 16).unwrap());
        end = st;
    }
    out
}

pub fn u64h(s: &str) -> u64 {
    let l = limbs(s);
    assert!(l.iter().skip(1).all(|&x| x == 0), "value exceeds u64");
    l[0]
}

pub fn container_bytes(v: &J) -> Vec<u8> {
    let mut b = unhex(v.get("prefix_hex").str());
    b.resize(v.get("total_len").num() as usize, 0);
    b
}
