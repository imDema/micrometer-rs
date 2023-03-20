use std::{
    collections::HashMap,
    ops::AddAssign,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use thread_local::ThreadLocal;

pub const BUFFER_INIT_BYTES: usize = 8 << 10; // 8kiB
pub const PERFORATION_STEP: usize = 1024;

static GLOBAL_REGISTRY: Lazy<Registry> = Lazy::new(Registry::default);

#[inline]
pub fn global() -> &'static Registry {
    &GLOBAL_REGISTRY
}

#[cfg(feature = "enable")]
pub fn summary_grouped() {
    let mut v: Vec<_> = global().stats_grouped().into_iter().collect();
    v.sort_unstable_by_key(|(name, _)| name.clone());
    v.into_iter().for_each(|(s, q)| {
        eprintln!(
            "{s:36}: {mean:12?}({std:12?}) {tot:12?} [{n:6}]({copies:2})",
            mean = q.mean,
            std = Duration::from_secs_f64(q.var.sqrt()),
            tot = q.sum,
            n = q.count,
            copies = q.copies,
        )
    })
}

#[cfg(feature = "enable")]
pub fn summary() {
    let mut v: Vec<_> = global().stats().into_iter().collect();
    v.sort_by_key(|(name, _)| name.clone());
    v.into_iter().for_each(|(s, q)| {
        eprintln!(
            "{s:36}: {mean:12?}({std:12?}) {tot:12?} [{n:6}]({copies:2})",
            mean = q.mean,
            std = Duration::from_secs_f64(q.var.sqrt()),
            tot = q.sum,
            n = q.count,
            copies = q.copies,
        )
    })
}

#[cfg(feature = "enable")]
pub fn save_csv(path: impl AsRef<Path>) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let mut f = File::create(path)?;
    let mut w = BufWriter::new(&mut f);
    let mut threads: HashMap<_, usize> = HashMap::new();

    let mut data = global().drain_raw();
    data.sort_by(|a, b| a.0.cmp(&b.0));

    writeln!(
        &mut w,
        "\"index\",\"name\",\"thread\",\"count\",\"time\",\"end\""
    )?;

    for (name, records) in data {
        let t = threads.entry(name.clone()).or_default();
        let thread = *t;
        *t += 1;
        let mut idx = 0;
        for Record { ns, count, end } in records {
            if !count.is_power_of_two() {
                continue;
            }
            let duration = Duration::from_nanos(ns) / count;
            let end = Duration::from_nanos(end).as_secs_f64();
            let seconds = duration.as_secs_f64();
            writeln!(
                &mut w,
                "\"{idx}\",\"{name}\",\"{thread}\",\"{count}\",\"{seconds:e}\",\"{end:e}\""
            )?;
            idx += count as usize;
        }
    }

    Ok(())
}

#[cfg(feature = "enable")]
pub fn append_csv(path: impl AsRef<Path>, experiment: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let mut f = File::options().append(true).create(true).open(path)?;
    if f.metadata()?.len() == 0 {
        writeln!(
            &mut f,
            "\"index\",\"experiment\",\"name\",\"thread\",\"count\",\"time\",\"end\""
        )?;
    }
    let mut w = BufWriter::new(&mut f);
    let mut threads: HashMap<_, usize> = HashMap::new();

    let mut data = global().drain_raw();
    data.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, records) in data {
        let t = threads.entry(name.clone()).or_default();
        let thread = *t;
        *t += 1;
        let mut idx = 0;
        for Record { ns, count, end } in records {
            if !count.is_power_of_two() {
                continue;
            }
            let duration = Duration::from_nanos(ns) / count;
            let end = Duration::from_nanos(end).as_secs_f64();
            let seconds = duration.as_secs_f64();
            writeln!(
                &mut w,
                "\"{idx}\",\"{experiment}\",\"{name}\",\"{thread}\",\"{count}\",\"{seconds:e}\",\"{end:e}\""
            )?;
            idx += count as usize;
        }
    }

    Ok(())
}

#[cfg(feature = "enable")]
pub fn save_csv_uniform(path: impl AsRef<Path>) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let mut f = File::create(path)?;
    let mut w = BufWriter::new(&mut f);
    let mut threads: HashMap<_, usize> = HashMap::new();

    let mut data = global().drain_raw();
    data.sort_by(|a, b| a.0.cmp(&b.0));

    writeln!(&mut w, "\"index\",\"name\",\"thread\",\"count\",\"time\"")?;

    for (name, records) in data {
        if records.is_empty() {
            continue;
        }
        let t = threads.entry(name.clone()).or_default();
        let thread = *t;
        *t += 1;

        let width = records
            .get(records.len() - 2)
            .unwrap_or_else(|| &records[records.len() - 1])
            .count;

        let mut r = Record::default();
        let mut rev_vec = Vec::new();
        for i in (0..records.len()).rev() {
            if i == records.len() - 1 && records[i].count != width {
                continue;
            }

            r += records[i];

            if r.count == width {
                rev_vec.push(r);
                r = Default::default();
            } else if r.count > width {
                eprintln!(
                    "WARN: micrometer illegal state {} ({}) {:+}. Resetting current",
                    r.count,
                    width,
                    r.count - width
                );
                r = Default::default();
            }
        }
        for (i, r) in rev_vec.into_iter().rev().enumerate() {
            let count = r.count;
            let duration = Duration::from_nanos(r.ns) / count;
            let seconds = duration.as_secs_f64();
            let index = i * width as usize;
            writeln!(
                &mut w,
                "\"{index}\",\"{name}\",\"{thread}\",\"{count}\",\"{seconds:e}\""
            )?;
        }
    }

    Ok(())
}

#[cfg(not(feature = "enable"))]
pub fn summary() {
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
}

#[cfg(not(feature = "enable"))]
pub fn summary_grouped() {
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
}

#[cfg(not(feature = "enable"))]
pub fn save_csv(path: impl AsRef<Path>) -> std::io::Result<()> {
    let _ = path;
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
    Ok(())
}

#[cfg(not(feature = "enable"))]
pub fn save_csv_uniform(path: impl AsRef<Path>) -> std::io::Result<()> {
    let _ = path;
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
    Ok(())
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
                (t.name.into(), std::mem::take(&mut g.vec))
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
                    .append(&mut std::mem::take(&mut g.vec));
                h
            })
    }

    pub fn stats_grouped(&self) -> HashMap<String, Stats> {
        #[derive(Default, Debug)]
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
                if !buf.vec.is_empty() {
                    let mut part = buf.vec.iter().fold(Part::default(), |mut part, r| {
                        part.count += r.count as usize;
                        part.sum += Duration::from_nanos(r.ns);
                        part
                    });
                    part.copies = 1;

                    *h.entry(name).or_default() += part;
                }
                h
            });

        let map = self
            .trackers
            .lock()
            .iter()
            .map(|t| (t.name, t.buf.0.lock()))
            .fold(map, |mut h, (name, buf)| {
                if !buf.vec.is_empty() {
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
                }
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
                        var: part.sqsum / (part.count - 1).max(1) as f64,
                    },
                )
            })
            .collect()
    }

    pub fn stats(&self) -> Vec<(String, Stats)> {
        self.trackers
            .lock()
            .iter()
            .map(|t| {
                let mut buf = t.buf.0.lock();
                buf.consolidate();
                let (sum, count): (Duration, usize) =
                    buf.vec.iter().fold(Default::default(), |(sum, count), r| {
                        (sum + Duration::from_nanos(r.ns), count + r.count as usize)
                    });
                let mean = (sum / count as u32).as_secs_f64();
                let sqsum = buf.vec.iter().fold(0., |acc, r| {
                    let s = r.ns as f64 / 1_000_000_000.;
                    let c = r.count as f64;
                    let d = (s / c) - mean;

                    acc + (d * d * c)
                });
                (
                    t.name.to_owned(),
                    Stats {
                        copies: 1,
                        count,
                        sum,
                        mean: sum / count as u32,
                        var: sqsum / (count - 1).max(1) as f64,
                    },
                )
            })
            .collect()
    }
}

#[cfg(feature = "instant")]
static START: Lazy<Instant> = Lazy::new(Instant::now);

#[cfg(feature = "instant")]
pub fn start() {
    Lazy::force(&START);
}

#[derive(Default, Clone, Copy)]
pub struct Record {
    pub ns: u64,
    pub count: u32,
    #[cfg(feature = "instant")]
    pub end: u64,
}

impl Record {
    #[inline]
    pub fn mean(&self) -> Duration {
        Duration::from_nanos(self.ns / self.count as u64)
    }
}

impl AddAssign<Record> for Record {
    fn add_assign(&mut self, rhs: Record) {
        self.ns += rhs.ns;
        self.count += rhs.count;
        self.end = self.end.max(rhs.end);
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
    #[inline]
    fn update(&mut self, d: Duration) {
        let q = self.vec.len() / PERFORATION_STEP;
        match q {
            0 => self.vec.push(Record {
                ns: d.as_nanos() as u64,
                count: 1,
                end: START.elapsed().as_nanos() as u64,
            }),
            q => {
                self.cur.count += 1;
                self.cur.ns += d.as_nanos() as u64;
                if self.cur.count == 1 << q {
                    self.cur.end = START.elapsed().as_nanos() as u64;
                    self.consolidate();
                }
            }
        }
    }

    #[cfg(not(feature = "perforation"))]
    #[inline]
    fn update(&mut self, d: Duration) {
        self.vec.push(Record {
            ns: d.as_nanos() as u64,
            count: 1,
        });
    }

    #[inline]
    fn consolidate(&mut self) {
        if self.cur.count > 0 {
            self.vec.push(std::mem::take(&mut self.cur))
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
            vec: Vec::with_capacity(BUFFER_INIT_BYTES / std::mem::size_of::<Record>()),
        })))
    }
}

pub struct DisabledSpan;
impl Drop for DisabledSpan {
    fn drop(&mut self) {}
}

pub struct TrackPoint(Lazy<ThreadLocal<Track>>);

impl TrackPoint {
    #[inline]
    pub const fn new_thread_local() -> Self {
        Self(Lazy::new(ThreadLocal::new))
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
    #[cfg(feature = "enable")]
    #[inline]
    pub fn span(&self) -> Span<'_> {
        Span::new(self)
    }

    #[cfg(not(feature = "enable"))]
    #[inline]
    pub fn span(&self) -> DisabledSpan {
        DisabledSpan
    }

    #[inline]
    pub fn record(&self, r: Duration) {
        self.buf.record(r);
    }
}

#[derive(Clone)]
pub struct Span<'a> {
    start: Instant,
    owner: &'a Track,
    armed: bool,
}

impl<'a> Span<'a> {
    #[cfg(feature = "enable")]
    #[inline]
    fn new(owner: &'a Track) -> Self {
        let start = Instant::now();
        Self {
            start,
            owner,
            armed: true,
        }
    }

    #[inline]
    pub fn arm(&mut self) {
        self.armed = true;
    }

    #[inline]
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for Span<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            self.owner.record(self.start.elapsed());
        }
    }
}

pub struct Stats {
    pub copies: usize,
    pub count: usize,
    pub sum: Duration,
    pub mean: Duration,
    pub var: f64,
}

#[cfg(feature = "enable")]
#[macro_export]
macro_rules! span {
    () => {
        $crate::span!(
            __span,
            concat!(module_path!(), ":", line!(), ":", column!())
        )
    };
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
    ($e:expr) => {{
        #[allow(unused)]
        let __span = {
            use $crate::TrackPoint;
            static TRACK_POINT: TrackPoint = TrackPoint::new();
            TRACK_POINT
                .get_or_init(concat!(module_path!(), ":", line!(), ":", column!()))
                .span()
        };
        $e
    }};
    ($e:expr, $name:expr) => {{
        #[allow(unused)]
        let __span = {
            use $crate::TrackPoint;
            static TRACK_POINT: TrackPoint = TrackPoint::new();
            TRACK_POINT.get_or_init($name).span()
        };
        $e
    }};
}

#[cfg(not(feature = "enable"))]
#[macro_export]
macro_rules! span {
    () => {
        $crate::DisabledSpan
    };
    ($span:ident) => {
        #[allow(unused)]
        let $span = $crate::DisabledSpan;
    };
    ($span:ident, $name:expr) => {
        #[allow(unused)]
        let $span = $crate::DisabledSpan;
    };
    ($e:expr) => {
        $e
    };
    ($e:expr, $name:expr) => {{
        $e
    }};
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
        println!();
        summary_grouped();
    }
}
