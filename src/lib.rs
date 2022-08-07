use std::{
    collections::HashMap,
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
    global().drain_raw_merged().into_iter().for_each(|(s, v)| {
        println!(
            "{s:20}: {avg:10?} {tot:12?} [{n:6}]",
            avg = v.iter().sum::<Duration>() / v.len() as u32,
            tot = v.iter().sum::<Duration>(),
            n = v.len()
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
            t.buf.0.lock().drain(..);
        });
    }

    pub fn drain_raw(&self) -> Vec<(String, Vec<Duration>)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| {
                (
                    t.name.into(),
                    std::mem::replace(t.buf.0.lock().as_mut(), Vec::new()),
                )
            })
            .collect()
    }

    pub fn drain_raw_merged(&self) -> HashMap<String, Vec<Duration>> {
        self.trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock()))
            .fold(HashMap::new(), |mut h, mut x| {
                h.entry(x.0)
                    .or_default()
                    .extend(std::mem::replace(x.1.as_mut(), Vec::new()).drain(..));
                h
            })
    }

    pub fn get_cnt_avg(&self) -> Vec<(String, usize, Duration)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| {
                let v = t.buf.0.lock();
                let len = v.len();
                (t.name.into(), len, v.iter().sum::<Duration>() / len as u32)
            })
            .collect()
    }
}

type Record = Duration;

#[derive(Clone)]
struct RecordBuf(Arc<Mutex<Vec<Record>>>);

impl Default for RecordBuf {
    #[inline]
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::with_capacity(
            (4 << 10) / std::mem::size_of::<Record>(),
        ))))
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
    pub fn record(&self, r: Record) {
        // Lock is never contended except on consultation of results
        // so despite the lock, this is cheap and non blocking most of the times
        self.buf.0.lock().push(r);
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
