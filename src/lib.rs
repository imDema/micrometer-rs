/*!
# Micrometer
Profiling for fast, high frequency events in multithreaded applications with low overhead

### Important

By default every measure is a no-op, to measure and consume measures, enable the `enable`
feature. This is done to allow libs to instrument their code without runitme costs if
no one is using the measures.

## Definitions

+ *Measurement*: a measurement is the timing data gathered for a single event.
+ *Record*: a record is the retrieval format for measurements. Records may be 1-to-1 with
respects to measurements, however, if *perforation* is enabled (default), as measurements
increase in numbers, a single record may be the aggregate of multiple measurements.
+ *Location*: a location is a region of interest of the program. All measurements start
from a location. Each measure is associated with the name of the location it originated
from.
+ *Thread location*: as `micrometer` is intended for multithreaded environments, each
*location* may be relevant for multiple threads, the location in the context of a specific
thread is referred to as *thread location*
+ `Span`: a *span* is a guard object that measures the time between when it was created
and when it was dropped (unless it was disarmed). Each span will produce one measurement
+ `Track`: a *track* is the struct responsible of handling all *measurements* originating
from a *thread location*. It is used to create spans or manually record the duration of an
event.
+ `TrackPoint`: while a *track* maps to the concept of *thread location* (as a single track
can only be owned by one thread), a *track point* maps to the concept of *location*. The
track point struct is a lazily allocated, thread local handle to multiple tracks that are
associated with the same name. Usually, track points are used as static variables, using
the `get_or_init` method to get the *track* for the active thread.

## Examples

#### Measuring the duration of a loop

```
for _ in 0..100 {
    // Define a `TrackPoint` named "loop_duration", get (or init) the `Track` for the
    // current thread, then create a span and assign it to a new local variable called
    // `loop_duration`
    micrometer::span!(loop_duration);

    std::hint::black_box("do something");
    // `loop_duration` is automatically dropped recording the measure
}
// Print a summary of the measurements
micrometer::summary();
```

#### Measuring the duration of a loop, threaded

```
std::thread::scope(|s| {
    for t in 1..=4 {
        s.spawn(move || {
            for _ in 0..(10 * t) {
                micrometer::span!(); // Name automatically assigned to source file and line
                std::hint::black_box("do something");
            }
        });
    }
});
// Print a summary of the measurements
micrometer::summary();
// Print a summary of the measurements, aggregating measures for the same location
micrometer::summary_grouped();
```
### Measuring the duration of an expression

```
// This works like the `dbg!` macro, allowing you to transparently wrap an expression:
// a span is created, the expression is executed, then the span is closed and the result
// of the expression is passed along
let a = micrometer::span!(5 * 5, "fives_sq");
let b = micrometer::span!(a * a); // Name automatically assigned to source file and line
assert_eq!(a, 25);
assert_eq!(b, 25 * 25);
```

### Measuring a code segment

```
let a = 5;
micrometer::span!(guard, "a_sq");
let b = a * a;
drop(guard); // Measurement stops here
let c = b * a;
```

*/

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    ops::AddAssign,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use thread_local::ThreadLocal;

/// Minimum size fo the buffer in which measurements are stored
pub const BUFFER_INIT_BYTES: usize = 8 << 10; // 8kiB

/// When using perforation (default) consequent measures will be combined
/// if the number of measures for a track point exceeds `PERFORATION_STEP`
///
/// The first `PERFORATION_STEP` measures will be saved as is, after that,
/// the following `PERFORATION_STEP` measures will be composed of 2 measures
/// each, the next will be 4 each and so on
pub const PERFORATION_STEP: usize = 1024;

static GLOBAL_REGISTRY: Lazy<Registry> = Lazy::new(Registry::default);

/// Get a reference to the global micrometer register
#[inline]
pub fn global() -> &'static Registry {
    &GLOBAL_REGISTRY
}

#[cfg(not(feature = "enable"))]
pub fn summary_grouped() {
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
}

#[cfg(feature = "enable")]
/// Print a summary for all track points. Group all measurements with the
/// same name.
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

#[cfg(not(feature = "enable"))]
pub fn summary() {
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
}

#[cfg(feature = "enable")]
/// Print a summary for all track points. Report measures with the same name,
/// but different thread separately
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

#[cfg(not(feature = "enable"))]
pub fn save_csv(path: impl AsRef<Path>) -> std::io::Result<()> {
    let _ = path;
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
    Ok(())
}

#[cfg(feature = "enable")]
/// Save all measurements to a csv file
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

#[cfg(not(feature = "enable"))]
#[allow(unused_variables)]
pub fn append_csv(path: impl AsRef<Path>, experiment: &str) -> std::io::Result<()> {
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
    Ok(())
}

#[cfg(feature = "enable")]
/// Append all measurements to a csv file
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

#[cfg(not(feature = "enable"))]
pub fn save_csv_uniform(path: impl AsRef<Path>) -> std::io::Result<()> {
    let _ = path;
    eprintln!("micrometer disabled, add 'enable' feature to gather statistics.");
    Ok(())
}

#[cfg(feature = "enable")]
/// Save all measurements to a csv file. Measures will be uniformed in count
/// to the largest granularity that has been measured for each location.
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
            .get(records.len().saturating_sub(2))
            .expect("records.is_empty() checked early, will always have at least 1 element")
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

/// Measurement registry, used to register new track points and collect results.
#[derive(Default)]
pub struct Registry {
    trackers: Mutex<Vec<Track>>,
}

impl Registry {
    /// Register a new track
    #[inline]
    pub fn register(&self, t: &Track) {
        self.trackers.lock().push(t.clone());
    }

    /// Delete all measurements for all track points
    pub fn clear(&self) {
        self.trackers.lock().iter().for_each(|t| {
            let mut g = t.buf.0.lock();
            g.consolidate();
            g.vec.drain(..);
        });
    }

    /// Get all the measurements for all tracks leaving them empty.
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

    /// Get all the measurements for all tracks leaving them empty.
    /// Merges tracks with the same name
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

    /// Get basic statistics about all tracks. Groups tracks with
    /// the same name
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

        let lock = self.trackers.lock();
        let mut locked_trackers = lock
            .iter()
            .map(|t| (t.name.to_string(), t.buf.0.lock()))
            .collect::<Vec<_>>();

        let map: HashMap<String, Part> =
            locked_trackers
                .iter_mut()
                .fold(HashMap::new(), |mut h, (name, buf)| {
                    buf.consolidate();
                    if !buf.vec.is_empty() {
                        let mut part = buf.vec.iter().fold(Part::default(), |mut part, r| {
                            part.count += r.count as usize;
                            part.sum += Duration::from_nanos(r.ns);
                            part
                        });
                        part.copies = 1;

                        *h.entry(name.clone()).or_default() += part;
                    }
                    h
                });

        let map = locked_trackers.iter().fold(map, |mut h, (name, buf)| {
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

    /// Get basic statistics about all tracks. Groups tracks with
    pub fn stats(&self) -> Vec<(String, Stats)> {
        self.trackers
            .lock()
            .iter()
            .filter_map(|t| {
                let mut buf = t.buf.0.lock();
                if buf.vec.is_empty() {
                    return None;
                }
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
                Some((
                    t.name.to_owned(),
                    Stats {
                        copies: 1,
                        count,
                        sum,
                        mean: sum / count as u32,
                        var: sqsum / (count - 1).max(1) as f64,
                    },
                ))
            })
            .collect()
    }
}

#[cfg(feature = "instant")]
static START: Lazy<Instant> = Lazy::new(Instant::now);

/// Force initialization of start time. If not called it will be
/// automatically initialized when the first measurement is recorded
#[cfg(feature = "instant")]
pub fn start() {
    Lazy::force(&START);
}

/// Single unit of measurement
#[derive(Default, Clone, Copy)]
pub struct Record {
    /// Total duration of recorded events
    pub ns: u64,
    /// Number of recorded events
    pub count: u32,
    /// End timestamp for the last event
    #[cfg(feature = "instant")]
    pub end: u64,
}

impl Record {
    /// Mean duration of recorded events
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

/// Lazy, thread local track. Should be used as a static variable
/// for each point of the program that needs to be measured.
///
/// ```
/// # use micrometer::*;
/// // Initialize a thread local track point
/// static TRACK_POINT: TrackPoint = TrackPoint::new_thread_local();
/// // Use the track point to create a new span
/// let span = TRACK_POINT.get_or_init("name").span();
/// ```
pub struct TrackPoint(Lazy<ThreadLocal<Track>>);

impl TrackPoint {
    /// Create a new thread local track point.
    #[inline]
    pub const fn new_thread_local() -> Self {
        Self(Lazy::new(ThreadLocal::new))
    }

    /// Get the thread local track that is used to measure events.
    /// The name of the track will be initialized to `name` on the first call (for each thread)
    /// and cannot be modified.
    #[inline]
    pub fn get_or_init(&self, name: &'static str) -> &Track {
        self.0.get_or(|| Track::new(name))
    }
}

/// Component used to create and save measures
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
    #[inline]
    pub fn span(&self) -> Span<'_> {
        Span::new(self)
    }

    #[inline]
    pub fn record(&self, r: Duration) {
        self.buf.record(r);
    }
}

#[derive(Clone)]
#[cfg(feature = "enable")]
pub struct Span<'a> {
    start: Instant,
    owner: &'a Track,
    armed: bool,
}

#[cfg(feature = "enable")]
impl<'a> Span<'a> {
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

#[cfg(feature = "enable")]
impl Drop for Span<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            self.owner.record(self.start.elapsed());
        }
    }
}

#[derive(Clone, Default)]
#[cfg(not(feature = "enable"))]
pub struct Span<'a> {
    owner: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "enable"))]
impl<'a> Span<'a> {
    #[inline]
    fn new(_owner: &'a Track) -> Self {
        Self {
            owner: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn arm(&mut self) {}

    #[inline]
    pub fn disarm(&mut self) {}
}

#[cfg(not(feature = "enable"))]
impl Drop for Span<'_> {
    #[inline]
    fn drop(&mut self) {}
}

#[derive(Debug)]
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
            static TRACK_POINT: TrackPoint = TrackPoint::new_thread_local();
            TRACK_POINT.get_or_init($name).span()
        };
    };
    ($e:expr) => {{
        #[allow(unused)]
        let __span = {
            use $crate::TrackPoint;
            static TRACK_POINT: TrackPoint = TrackPoint::new_thread_local();
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
            static TRACK_POINT: TrackPoint = TrackPoint::new_thread_local();
            TRACK_POINT.get_or_init($name).span()
        };
        $e
    }};
}

#[cfg(not(feature = "enable"))]
#[macro_export]
macro_rules! span {
    () => {
        $crate::Span::default()
    };
    ($span:ident) => {
        #[allow(unused)]
        let $span = $crate::Span::default();
    };
    ($span:ident, $name:expr) => {
        #[allow(unused)]
        let $span = $crate::Span::default();
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

    fn example_events() {
        super::span!(once_event);
        for _ in 0..50 {
            super::span!(loop_event);
            std::hint::black_box("do something");
            _ = super::span!(5 * 5, "expr_event");
        }
    }

    #[test]
    fn stats() {
        example_events();
        eprintln!("{:?}", super::global().stats());
        super::global().clear();
    }

    #[test]
    fn stats_grouped() {
        example_events();
        eprintln!("{:?}", super::global().stats_grouped());
        super::global().clear();
    }

    #[test]
    fn stats_threaded() {
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(example_events);
            }
        });
        eprintln!("{:?}", super::global().stats());
        super::global().clear();
    }

    #[test]
    fn stats_grouped_threaded() {
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(example_events);
            }
        });
        eprintln!("{:?}", super::global().stats_grouped());
        super::global().clear();
    }

    #[test]
    fn save_csv() {
        example_events();

        let dir = tempfile::tempdir().expect("failed to create temp file");
        let path = dir.path().join("micrometer-test.csv");
        super::save_csv(path).expect("failed to save csv");
        let path = dir.path().join("micrometer-test-uniform.csv");
        super::save_csv_uniform(path).expect("failed to save csv");
    }

    #[test]
    fn append_csv() {
        example_events();

        let dir = tempfile::tempdir().expect("failed to create temp file");
        let path = dir.path().join("micrometer-test.csv");
        super::append_csv(&path, "a").expect("failed to save csv");
        super::append_csv(&path, "b").expect("failed to save csv");
    }

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
