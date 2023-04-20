# Micrometer

[![Crates.io](https://img.shields.io/crates/v/micrometer.svg)](https://crates.io/crates/micrometer)
[![Documentation](https://docs.rs/micrometer/badge.svg)](https://docs.rs/micrometer)

Profiling for fast, high frequency events in multithreaded applications with low overhead

### Important: enabling data collection

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

```rs
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

```rs
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

```rs
// This works like the `dbg!` macro, allowing you to transparently wrap an expression:
// a span is created, the expression is executed, then the span is closed and the result
// of the expression is passed along
let a = micrometer::span!(5 * 5, "fives_sq");
let b = micrometer::span!(a * a); // Name automatically assigned to source file and line
assert_eq!(a, 25);
assert_eq!(b, 25 * 25);
```

### Measuring a code segment

```rs
let a = 5;
micrometer::span!(guard, "a_sq");
let b = a * a;
drop(guard); // Measurement stops here
let c = b * a;
```