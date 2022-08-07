use std::{
    sync::Arc,
    time::{Duration, Instant}, collections::HashMap,
};

pub use once_cell::sync::Lazy;
use parking_lot::Mutex;
pub use thread_local::ThreadLocal;

static GLOBAL_REGISTRY: Lazy<Registry> = Lazy::new(|| Registry::default());

#[derive(Default)]
pub struct Registry {
    trackers: Mutex<Vec<Track>>,
}

pub fn global() -> &'static Registry {
    &GLOBAL_REGISTRY
}

impl Registry {
    pub fn register(&self, t: &Track) {
        self.trackers.lock().push(t.clone());
    }

    pub fn clear(&self) {
        self.trackers
            .lock()
            .iter()
            .for_each(|t| { t.buf.0.lock().drain(..); });
    }

    pub fn get_raw(&self) -> Vec<(String, Vec<Duration>)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock().clone()))
            .collect()
    }

    pub fn get_raw_merged(&self) -> HashMap<String, Vec<Duration>> {
        self.trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock().clone()))
            .fold(HashMap::new(), |mut h, x| {
                h.entry(x.0).or_default().extend(x.1);
                h
            })
    }

    pub fn get_cnt_avg(&self) -> Vec<(String, usize, Duration)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| (t.name.into(), t.buf.0.lock().clone()))
            .map(|(n, v)|  {
                let len = v.len();
                (n, len, v.into_iter().sum::<Duration>() / len as u32)
            })
            .collect()
    }
}

type Record = Duration;

#[derive(Clone)]
struct RecordBuf(Arc<Mutex<Vec<Record>>>);

impl Default for RecordBuf {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Clone)]
pub struct Track {
    name: &'static str,
    buf: RecordBuf,
}

impl Track {
    pub fn span(&self) -> Span<'_> {
        Span::new(self)
    }

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

#[macro_export]
macro_rules! span {
    ($span:ident) => {
        $crate::span!($span, stringify!($span))
    };
    ($span:ident, $name:expr) => {
        #[allow(unused)]
        let $span = {
            use $crate::{Lazy, ThreadLocal, Track};

            static TRACK_POINT: Lazy<ThreadLocal<Track>> = Lazy::new(|| ThreadLocal::new());
            // static SERIAL: AtomicUsize = AtomicUsize::new(0);

            let t = TRACK_POINT.get_or(|| {
                let buf = RecordBuf::default();
                let name = $name;
                let t = Track { buf, name };

                global().register(&t);

                t
            });

            t.span()
        };
    };
}

pub fn summary() {
    global()
        .get_raw_merged()
        .into_iter()
        .for_each(|(s, v)| println!(
            "{s:20}: {d:12?} [{n:6}]",
            d = v.iter().sum::<Duration>() / v.len() as u32,
            n = v.len()
        ))
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
