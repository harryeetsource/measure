#![feature(asm)]
#![allow(deprecated)] // To allow the use of inline assembly, which is currently unstable
use core::arch::asm;
use std::{sync::{Arc, Mutex}, thread, time::Duration};
use std::process;
use core::mem;
use std::sync::mpsc::channel;
/// Reads the current value of the processor's time-stamp counter.
fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem, preserves_flags),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Performs a series of simple operations.
fn simple_operations(iterations: u32) {
    let mut arr = [1, 2, 3, 4, 5];
    let mut x = 0;
    let mut pid: u32;
    let mut data2 = vec![0u32; 1024]; // For memory access pattern
    let mut result = 0u64;
    let mut data = vec![0u64; 1024]; 

    // Advanced Computational Patterns
    for i in 0..iterations {
        // SIMD-like operation simulation
        result += data.iter().enumerate().map(|(idx, &val)| val.wrapping_add(idx as u64 * i as u64)).sum::<u64>();

        // Floating-point operations and dynamic workload adjustment
        if i % 100 == 0 {
            result += (result as f64 * 1.01).floor() as u64;
        }

        // Intentionally cause cache misses
        if i % 10 == 0 {
            data.rotate_left(1);
        }
    

    // Prevent optimization
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    
        // Arithmetic operations
        x = i.wrapping_add(1); // Addition
        x = x.wrapping_mul(3); // Multiplication
        if x % 2 == 0 { // Conditional to prevent optimization
            x = x.wrapping_div(2); // Division, avoid division by zero
        }
        x = x.wrapping_sub(i); // Subtraction

        // Bitwise operations
        x = x ^ 0x55; // XOR
        x = x >> 2; // Right shift
        x = x << 4; // Left shift
        x = x | 1; // OR
        x = x & 1; // AND
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); 
        // Memory operations
        unsafe {
            std::ptr::write_volatile(&mut x, i); // Write
            x = std::ptr::read_volatile(&x); // Read
        }

        // Array access
        x = arr[(i as usize) % arr.len()]; // Array read based on i

        // Control flow
        let mut j = 0;
        loop {
            j += 1;
            if j > 5 { break; } // Simple loop with a break condition
        }

        // Function call
        x = simple_function(x); // Function that can be inlined
     // System call overhead: Getting the process ID
     pid = process::id();
     mem::forget(pid); // Ensure the PID call is not optimized away

    

     // Memory access pattern: Random accesses within an array
     let index = (i.wrapping_mul(997)) as usize % data2.len(); // Pseudo-random index
     data2[index] = pid; // Use PID as data to prevent optimization
     x = data2[index];

     // Floating-point operations
     let fop = (i as f64) * 0.5;
     mem::forget(fop); // Use floating-point operation in a minimal way
 }
 // Use x to ensure computations are not optimized away
 std::mem::forget(x);
 std::mem::forget(data);
 std::mem::forget(result);
 core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); // Ensure memory pattern operations are not optimized away
}

fn simple_function(input: u32) -> u32 {
    // A simple inlinable function that does a basic computation
    input.wrapping_mul(2) // Just an example operation
}


fn main() {
    const ITERATIONS: u32 = 1000;
    const NUM_EXECUTIONS: u32 = 5000;
    const SLEEP_DURATION_MS_MAIN: u64 = 10; // Main thread sleep duration
    const SLEEP_DURATION_MS_SPAWNED: u64 = 1; // Spawned thread sleep duratio
    const MAX_ATTEMPTS: u32 = 2; // Allows for one retry
    const CYCLE_THRESHOLD: u64 = 31000;
    let mutex = Arc::new(Mutex::new(0)); // Shared mutex
    let (tx, rx) = channel(); // Channel for communicating cycles

    let mut attempt = 0;
    while attempt < MAX_ATTEMPTS {
        attempt += 1;
        let mutex_clone = Arc::clone(&mutex);
        let tx_clone = tx.clone();

        thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap(); // Lock the mutex in the spawned thread
            // Measure before performing operations
            let start = rdtsc();
            simple_operations(ITERATIONS); // Perform the workload
            let end = rdtsc();
            // Send the measured cycles back to the main thread
            tx_clone.send(end - start).unwrap();
        });

        // Main thread sleeps to simulate delay and ensure the spawned thread executes
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS_MAIN));
        // Try to acquire the mutex, forcing the main thread to wait until the spawned thread releases it
        let _lock = mutex.lock().unwrap();

        // Receive the cycle count from the spawned thread
        if let Ok(cycles) = rx.recv() {
            println!("Attempt {}: Cycles for simple operations: {}", attempt, cycles);
            if cycles < CYCLE_THRESHOLD {
                println!("Below cycle threshold, indicating non-VM or efficient VM.");
                break; // Exit if we meet the condition
            }
        } else {
            println!("Error receiving cycles from thread.");
        }

        if attempt >= MAX_ATTEMPTS {
            println!("Attempt limit reached. Potential VM detected or inefficient VM.");
        }
    }
}