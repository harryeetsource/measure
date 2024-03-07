#![feature(asm)]
#![allow(deprecated)] // To allow the use of inline assembly, which is currently unstable
use core::arch::asm;
use std::{sync::{Arc, Mutex}, thread, time::Duration};
use std::process;
use core::mem;
use std::sync::mpsc::channel;
use std::collections::VecDeque;
use std::process::exit;
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
fn basic_cpu_operations(iterations: u32) {
    let mut y = 0u64; // Use a different variable for operations
    let mut sum = 0u64; // To accumulate results and prevent optimization
    let mut toggle = false;

    for i in 0..iterations {
        // Simple arithmetic operations with a twist
        y = y.wrapping_add(i as u64 * 3);
        y = y.wrapping_mul(2);

        // Introduce a conditional operation to simulate control flow
        if y % 4 == 0 {
            y = y.wrapping_div(2);
        } else {
            y = y.wrapping_mul(2);
        }

        // Bitwise operations with variability
        y = y ^ 0xAA; // XOR with a different constant
        y = y.rotate_left(3); // Use rotate to introduce variability
        y = y | 0x55;
        y = y & 0xFF;

        // Toggle operation to simulate dynamic behavior
        if toggle {
            y = !y; // Bitwise NOT every other iteration
        }
        toggle = !toggle;

        // Accumulate sum to use the result and prevent optimization
        sum = sum.wrapping_add(y);

        // Memory barrier to ensure operations are not optimized away
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }

    // Use sum in a trivial way to ensure it's not optimized out
    if sum == u64::MAX {
        println!("Unlikely event to use sum and prevent optimization: {}", sum);
    }
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


fn calculate_statistics(data: &VecDeque<u64>) -> (f64, f64) {
    let sum: u64 = data.iter().sum();
    let count = data.len() as f64;
    let mean = sum as f64 / count;

    let variance = data.iter().map(|value| {
        let diff = *value as f64 - mean;
        diff * diff
    }).sum::<f64>() / count;

    let std_deviation = variance.sqrt();

    (mean, std_deviation)
}

fn main() {
    const ITERATIONS: u32 = 1000;
    const MAX_ATTEMPTS: u32 = 50;
    const CYCLE_THRESHOLD: f64 = 31000.0;
    const CYCLE_THRESHOLD2: f64 = 10000.0;
    let mut cycles_data_simple = VecDeque::new();
    let mut cycles_data_basic = VecDeque::new();

    let mutex = Arc::new(Mutex::new(0));
    let (tx, rx) = channel();

    let mut successful_attempt = false; // To track if the loop exits due to condition being met

    for attempt in 0..MAX_ATTEMPTS {
        let mutex_clone_simple = Arc::clone(&mutex);
        let tx_clone_simple = tx.clone();

        let handle_simple = thread::spawn(move || {
            let _guard = mutex_clone_simple.lock().unwrap();
            let start = rdtsc();
            simple_operations(ITERATIONS);
            let end = rdtsc();
            tx_clone_simple.send(("simple", end - start)).unwrap();
        });

        let mutex_clone_basic = Arc::clone(&mutex);
        let tx_clone_basic = tx.clone();

        let handle_basic = thread::spawn(move || {
            let _guard = mutex_clone_basic.lock().unwrap();
            let start = rdtsc();
            basic_cpu_operations(ITERATIONS);
            let end = rdtsc();
            tx_clone_basic.send(("basic", end - start)).unwrap();
        });

        handle_simple.join().unwrap();
        handle_basic.join().unwrap();

        for _ in 0..2 {
            match rx.recv() {
                Ok(("simple", cycles)) => cycles_data_simple.push_back(cycles),
                Ok(("basic", cycles)) => cycles_data_basic.push_back(cycles),
                Ok(_) => (),
                Err(e) => println!("Error receiving cycles from thread: {:?}", e),
            }
        }

        let (mean_simple, std_deviation_simple) = calculate_statistics(&cycles_data_simple);
        let (mean_basic, std_deviation_basic) = calculate_statistics(&cycles_data_basic);

        println!("Attempt {}: Simple Mean cycles: {}, Standard Deviation: {}", attempt + 1, mean_simple, std_deviation_simple);
        println!("Attempt {}: Basic Mean cycles: {}, Standard Deviation: {}", attempt + 1, mean_basic, std_deviation_basic);

        if mean_simple < CYCLE_THRESHOLD && mean_basic < CYCLE_THRESHOLD2 {
            println!("Below cycle threshold, indicating non-VM or efficient VM.");
            successful_attempt = true; // Mark successful attempt
            break;
        }
    }

    // Check if the loop completed due to reaching the max attempts without satisfying condition
    if !successful_attempt {
        println!("Attempt limit reached. Potential VM detected or inefficient VM. Exiting program.");
        exit(1);
    }
}
