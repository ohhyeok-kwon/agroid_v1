//! Bounded single-producer/single-consumer handoff. Full queues drop the new item; no waits.
use std::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Queue<T: Copy, const N: usize> {
    slots: [UnsafeCell<MaybeUninit<T>>; N],
    read: AtomicUsize,
    write: AtomicUsize,
}
// A slot belongs to the producer until the release-store to write, then to the consumer
// until the release-store to read. Handles are unique, !Sync, and operations require &mut.
unsafe impl<T: Copy + Send, const N: usize> Sync for Queue<T, N> {}

pub struct Producer<T: Copy, const N: usize>(Arc<Queue<T, N>>, PhantomData<Cell<()>>);
pub struct Consumer<T: Copy, const N: usize>(Arc<Queue<T, N>>, PhantomData<Cell<()>>);

pub fn channel<T: Copy + Send, const N: usize>() -> (Producer<T, N>, Consumer<T, N>) {
    assert!(N >= 2);
    let q = Arc::new(Queue {
        slots: std::array::from_fn(|_| UnsafeCell::new(MaybeUninit::uninit())),
        read: AtomicUsize::new(0),
        write: AtomicUsize::new(0),
    });
    (
        Producer(Arc::clone(&q), PhantomData),
        Consumer(q, PhantomData),
    )
}
impl<T: Copy, const N: usize> Producer<T, N> {
    pub fn push(&mut self, value: T) -> bool {
        let w = self.0.write.load(Ordering::Relaxed);
        let next = (w + 1) % N;
        if next == self.0.read.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: this producer exclusively owns the free slot.
        unsafe {
            (*self.0.slots[w].get()).write(value);
        }
        self.0.write.store(next, Ordering::Release);
        true
    }
}
impl<T: Copy, const N: usize> Consumer<T, N> {
    pub fn pop(&mut self) -> Option<T> {
        let r = self.0.read.load(Ordering::Relaxed);
        if r == self.0.write.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: acquire observed a fully initialized slot; producer cannot reuse it yet.
        let value = unsafe { (*self.0.slots[r].get()).assume_init_read() };
        self.0.read.store((r + 1) % N, Ordering::Release);
        Some(value)
    }
    pub fn latest(&mut self) -> Option<T> {
        let mut value = None;
        // Bound the work even when the producer is active during this call.
        for _ in 0..N - 1 {
            match self.pop() {
                Some(v) => value = Some(v),
                None => break,
            }
        }
        value
    }
}
