use std::{
    collections::HashMap,
    ops::AddAssign,
    sync::Arc,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use thread_local::ThreadLocal;

static GLOBAL_REGISTRY: Lazy<Registry> = Lazy::new(|| Registry::default());

#[inline]
pub fn global() -> &'static Registry {
    &GLOBAL_REGISTRY
}

pub fn summary() {
    global().stats().into_iter().for_each(|(s, q)| {
        println!(
            "{s:20}: {mean:10?}({std:10?}) {tot:12?} [{n:6}]({copies:2})",
            mean = q.mean,
            std = Duration::from_secs_f32(q.var.as_secs_f32().sqrt()),
            tot = q.sum,
            n = q.count,
            copies = q.copies,
        )
    })
}

#[derive(Default)]
pub struct Registry {
    trackers: Mutex<Vec<Track>>,
}

impl Registry {
    #[inline]
    pub fn register(&self, t: &Track) {
        self.trackers.lock().push(t.clone());
    }

    pub fn clear(&self) {
        self.trackers.lock().iter().for_each(|t| {
            let mut g = t.buf.0.lock();
            g.consolidate();
            g.vec.drain(..);
        });
    }

    pub fn drain_raw(&self) -> Vec<(String, Vec<Record>)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| {
                let mut g = t.buf.0.lock();
                g.consolidate();
                (t.name.into(), std::mem::replace(&mut g.vec, Vec::new()))
            })
            .collect()
    }

    pub fn drain_raw_merged(&self) -> HashMap<String, Vec<Record>> {
        self.trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock()))
            .fold(HashMap::new(), |mut h, (name, mut g)| {
                g.consolidate();
                h.entry(name)
                    .or_default()
                    .extend(std::mem::replace(&mut g.vec, Vec::new()).drain(..));
                h
            })
    }

    pub fn stats(&self) -> HashMap<String, Stats> {
        #[derive(Default)]
        struct Part {
            copies: usize,
            count: usize,
            sum: Duration,
            sqsum: f64,
        }

        impl AddAssign<Part> for Part {
            fn add_assign(&mut self, rhs: Part) {
                self.copies += rhs.copies;
                self.count += rhs.count;
                self.sum += rhs.sum;
                self.sqsum += rhs.sqsum;
            }
        }

        let map: HashMap<String, Part> = self
            .trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock()))
            .fold(HashMap::new(), |mut h, (name, mut buf)| {
                buf.consolidate();
                let mut part = buf.vec.iter().fold(Part::default(), |mut part, r| {
                    part.count += r.count as usize;
                    part.sum += Duration::from_nanos(r.ns);
                    part
                });
                part.copies = 1;

                *h.entry(name).or_default() += part;
                h
            });

        let map = self
            .trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock()))
            .fold(map, |mut h, (name, buf)| {
                let mean = h
                    .get(name)
                    .map(|p| p.sum / p.count as u32)
                    .unwrap()
                    .as_secs_f64();
                let sqsum = buf.vec.iter().fold(0., |acc, r| {
                    let s = r.ns as f64 / 1_000_000_000.;
                    let c = r.count as f64;
                    let d = (s / c) - mean;
                    
                    acc + (d * d * c)
                });

                h.get_mut(name).unwrap().sqsum += sqsum;
                h
            });

        map.into_iter()
            .map(|(name, part)| {
                (
                    name,
                    Stats {
                        copies: part.copies,
                        count: part.count,
                        sum: part.sum,
                        mean: part.sum / part.count as u32,
                        var: Duration::from_secs_f64(part.sqsum / part.count as f64),
                    },
                )
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Record {
    pub ns: u64,
    pub count: u32,
}

impl Record {
    #[inline]
    pub fn avg(&self) -> Duration {
        Duration::from_nanos(self.ns / self.count as u64)
    }
}

#[derive(Clone)]
struct RecordBuf(Arc<Mutex<RecordBufInner>>);
struct RecordBufInner {
    cur: Record,
    vec: Vec<Record>,
}

impl RecordBufInner {
    #[cfg(feature = "perforation")]
    fn update(&mut self, d: Duration) {
        let q = self.vec.len() as u32 / 1024;
        match q {
            0 => self.vec.push(Record {
                ns: d.as_nanos() as u64,
                count: 1,
            }),
            q => {
                self.cur.count += 1;
                self.cur.ns += d.as_nanos() as u64;
                if self.cur.count == 1 << q {
                    self.consolidate();
                }
            }
        }
    }

    #[cfg(not(feature = "perforation"))]
    fn update(&mut self, d: Duration) {
        self.vec.push(Record {
            ns: d.as_nanos() as u64,
            count: 1,
        });
    }

    fn consolidate(&mut self) {
        if self.cur.count > 0 {
            self.vec
                .push(std::mem::replace(&mut self.cur, Default::default()))
        }
    }
}

impl RecordBuf {
    #[inline]
    fn record(&self, d: Duration) {
        // Lock is never contended except on consultation of results
        // so despite the lock, this is cheap and non blocking most of the times
        self.0.lock().update(d);
    }
}

impl Default for RecordBuf {
    #[inline]
    fn default() -> Self {
        Self(Arc::new(Mutex::new(RecordBufInner {
            cur: Default::default(),
            vec: Vec::with_capacity((4 << 10) / std::mem::size_of::<Record>()),
        })))
    }
}

pub struct TrackPoint(Lazy<ThreadLocal<Track>>);

impl TrackPoint {
    #[inline]
    pub const fn new() -> Self {
        Self(Lazy::new(|| ThreadLocal::new()))
    }

    #[inline]
    pub fn get_or_init(&self, name: &'static str) -> &Track {
        self.0.get_or(|| Track::new(name))
    }
}

#[derive(Clone)]
pub struct Track {
    name: &'static str,
    buf: RecordBuf,
}

impl Track {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let buf = RecordBuf::default();
        let t = Self { buf, name };

        global().register(&t);

        t
    }

    // Get a span guard, measures time from its creation to when it's dropped
    #[cfg(not(feature = "disable"))]
    #[inline]
    pub fn span(&self) -> Span<'_> {
        Span::new(self)
    }

    #[inline]
    #[cfg(feature = "disable")]
    pub fn span(&self) -> () {
        ()
    }

    #[inline]
    pub fn record(&self, r: Duration) {
        self.buf.record(r);
    }
}

pub struct Span<'a> {
    start: Instant,
    owner: &'a Track,
}

impl<'a> Span<'a> {
    fn new(owner: &'a Track) -> Self {
        let start = Instant::now();
        Self { start, owner }
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        self.owner.record(self.start.elapsed());
    }
}

pub struct Stats {
    pub copies: usize,
    pub count: usize,
    pub sum: Duration,
    pub mean: Duration,
    pub var: Duration,
}

#[cfg(not(feature = "disable"))]
#[macro_export]
macro_rules! span {
    ($span:ident) => {
        $crate::span!($span, stringify!($span))
    };
    ($span:ident, $name:expr) => {
        #[allow(unused)]
        let $span = {
            use $crate::TrackPoint;

            static TRACK_POINT: TrackPoint = TrackPoint::new();

            TRACK_POINT.get_or_init($name).span()
        };
    };
}

#[cfg(feature = "disable")]
#[macro_export]
macro_rules! span {
    ($span:ident) => {
        $crate::span!($span, stringify!($span))
    };
    ($span:ident, $name:expr) => {
        #[allow(unused)]
        let $span = ();
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn example() {
        span!(full);

        (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    span!(thread);
                    std::thread::sleep(Duration::from_micros(200));
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|j| j.join().unwrap());

        for _ in 0..16 {
            span!(inside);
            std::thread::sleep(Duration::from_micros(100));
        }

        drop(full);

        summary();
    }
}
