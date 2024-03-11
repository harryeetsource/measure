use core::arch::asm;
use std::{sync::{Arc, Mutex}, thread, time::Duration};
use std::process;
use core::mem;
use std::sync::mpsc::channel;
use std::collections::VecDeque;
use std::process::exit;
use std::fs::{OpenOptions};
use std::io::{Write, BufReader, BufRead};
use std::io;
use std::fs::File;
use std::path::Path;
use serde_derive::Deserialize;
use serde_json::Value;
use std::io::BufWriter;
use core::convert::From;
use smartcore::{
    ensemble::random_forest_classifier::RandomForestClassifier,
    metrics::accuracy,
    model_selection::train_test_split,
};
use serde_derive::Serialize;
use serde_json::json;
struct CycleDataStats {
    cycle_counts: Vec<u64>,
    mean_cycles: f64,
    std_dev_cycles: f64,
    max_cycle: u64,
    min_cycle: u64,
    label: u32,
}
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


fn read_cycle_data_from_file(file_path: &str) -> io::Result<VecDeque<u64>> {
    // Check if the file exists
    if !Path::new(file_path).exists() {
        // If the file does not exist, return an empty VecDeque<u64>
        return Ok(VecDeque::new());
    }

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut data = VecDeque::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(number) = line.parse::<u64>() {
            data.push_back(number);
        }
    }

    Ok(data)
}

fn append_cycle_data_to_file(file_path: &str, cycle_data: &[u64]) -> io::Result<()> {
    let mut file = OpenOptions::new().append(true).create(true).open(file_path)?;
    for &cycles in cycle_data {
        writeln!(file, "{}", cycles)?;
    }
    Ok(())
}
fn serialize_cycle_data(data: &[CycleMeasurement]) -> io::Result<()> {
    let json_data = serde_json::to_value(data)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let serialized = serde_json::to_string(&json_data)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let mut file = File::create("cycle_data.json")?;
    writeln!(file, "{}", serialized)?;
    Ok(())
}
#[derive(Serialize, Deserialize)]
struct CycleMeasurement {
    mean_cycles: u64,
    std_dev_cycles: f64,
    max_cycles: u64,
    label: u32, // 1 for VM, 0 for non-VM
}
fn calculate_statistics(cycle_counts: &Vec<u64>) -> (f64, f64, u64, u64) {
    let sum: u64 = cycle_counts.iter().sum();
    let count = cycle_counts.len() as f64;
    let mean = sum as f64 / count;
    let variance = cycle_counts.iter().map(|&value| {
        let diff = value as f64 - mean;
        diff * diff
    }).sum::<f64>() / count;
    let std_deviation = variance.sqrt();
    let max_cycle = *cycle_counts.iter().max().unwrap_or(&0);
    let min_cycle = *cycle_counts.iter().min().unwrap_or(&0);

    (mean, std_deviation, max_cycle, min_cycle)
}
unsafe fn rdtscp() -> (u64, u32) {
    let lo: u32;
    let hi: u32;
    let aux: u32;
    asm!(
        "rdtscp",
        out("eax") lo, out("edx") hi, lateout("ecx") aux,
        options(nomem, nostack, preserves_flags),
    );
    (((hi as u64) << 32) | lo as u64, aux)
}
// Function to load data from a JSON file
fn load_data_from_file<CycleData>(file_path: &str) -> io::Result<Vec<CycleData>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let data: Vec<CycleData> = serde_json::from_reader(reader)?;
    Ok(data)
}
fn prepare_dataset_for_training(data: Vec<CycleDataStats>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let features: Vec<Vec<f64>> = data.into_iter().map(|d| vec![
        d.mean_cycles, 
        d.std_dev_cycles, 
        d.max_cycle as f64, 
        d.min_cycle as f64
    ]).collect();
    let labels: Vec<f64> = data.iter().map(|d| d.label as f64).collect();

    (features, labels)
}
fn main() -> io::Result<()> {
    let data = collect_and_analyze_cycles(5000, 1)?;

    // Step 2: Preprocess the data into features and labels
    let (features, labels) = prepare_dataset_for_training(vec![data]);


    // Step 3: Train the model
    let dataset = (features, labels);
    let (train, test, train_labels, test_labels) = train_test_split(&features, &labels, 0.3, true, None);
    let model = RandomForestClassifier::fit(&train, &train_labels, Default::default())
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;


    // Step 4: Evaluate the model
    let predictions = model.predict(&test)?;
    let accuracy_rating = accuracy(&test.target, &predictions);
    println!("Model accuracy: {}", accuracy_rating);

    // Step 5: Serialize the model
    let model_file = File::create("model.json")?;
    let writer = BufWriter::new(model_file);
    serde_json::to_writer(writer, &model)?;

    Ok(())
}


fn collect_and_analyze_cycles(iterations: u32, label: u32) -> io::Result<CycleDataStats> {
    const DATA_FILE_PATH: &str = "cycle_data.txt";
    const ITERATIONS: u32 = 1000; // Ensure this is set as needed
    const MAX_ATTEMPTS: u32 = 5000;
    let mut historical_cycles_data = read_cycle_data_from_file(DATA_FILE_PATH)?;
    // Initialize cycle data vectors for this session
    let mut cycles_data_simple = VecDeque::new();
    let mut cycles_data_basic = VecDeque::new();



    let mutex = Arc::new(Mutex::new(0));
    let (tx, rx) = channel();
    let mut attempt = 0;
    let mut successful_attempt = false;

    while attempt < MAX_ATTEMPTS {
        attempt += 1;
        let mutex_clone_simple = Arc::clone(&mutex);
        let tx_clone_simple = tx.clone();
    
        // Spawn the first thread with RDTSC measurement
        let handle_simple = thread::spawn(move || {
            let _guard = mutex_clone_simple.lock().unwrap();
            let start = rdtsc();
            simple_operations(ITERATIONS);
            let end = rdtsc();
            tx_clone_simple.send(("simple", end - start)).unwrap();
        });
    
        // Clone `mutex` and `tx` again for the second use
        let mutex_clone_simple_rdtscp = Arc::clone(&mutex);
        let tx_clone_simple_rdtscp = tx.clone();
    
        // Spawn the second thread with RDTSCP measurement
        let handle_simple_rdtscp = thread::spawn(move || {
            let _guard = mutex_clone_simple_rdtscp.lock().unwrap();
            let (start, start_aux) = unsafe { rdtscp() };
            simple_operations(ITERATIONS);
            let (end, end_aux) = unsafe { rdtscp() };
            println!("Start AUX (simple operations): {}, End AUX: {}", start_aux, end_aux);
            tx_clone_simple_rdtscp.send(("simple_rdtscp", end - start)).unwrap();
        });
    
        // Similar cloning for "basic" operations
        let mutex_clone_basic = Arc::clone(&mutex);
        let tx_clone_basic = tx.clone();
    
        let handle_basic = thread::spawn(move || {
            let _guard = mutex_clone_basic.lock().unwrap();
            let start = rdtsc();
            basic_cpu_operations(ITERATIONS);
            let end = rdtsc();
            tx_clone_basic.send(("basic", end - start)).unwrap();
        });
    
        let mutex_clone_basic_rdtscp = Arc::clone(&mutex);
        let tx_clone_basic_rdtscp = tx.clone();
    
        let handle_basic_rdtscp = thread::spawn(move || {
            let _guard = mutex_clone_basic_rdtscp.lock().unwrap();
            let (start, start_aux) = unsafe { rdtscp() };
            basic_cpu_operations(ITERATIONS);
            let (end, end_aux) = unsafe { rdtscp() };
            println!("Start AUX (basic operations): {}, End AUX: {}", start_aux, end_aux);
            tx_clone_basic_rdtscp.send(("basic_rdtscp", end - start)).unwrap();
        });
    
        // Ensure all threads have completed
        handle_simple.join().unwrap();
        handle_simple_rdtscp.join().unwrap();
        handle_basic.join().unwrap();
        handle_basic_rdtscp.join().unwrap();
        for _ in 0..4 {
            match rx.recv() {
                Ok(("simple", cycles)) | Ok(("simple_rdtscp", cycles)) => cycles_data_simple.push_back(cycles),
                Ok(("basic", cycles)) | Ok(("basic_rdtscp", cycles)) => cycles_data_basic.push_back(cycles),
                Ok(_) => (),
                Err(e) => println!("Error receiving cycles from thread: {:?}", e),
            }
        }
        // Update historical data with new data for lifetime analysis
        historical_cycles_data.extend(cycles_data_simple.iter().cloned());
        historical_cycles_data.extend(cycles_data_basic.iter().cloned());
        
        // Ensure VecDeque data is correctly converted to Vec for statistical calculation.
    let historical_cycles_data_vec: Vec<u64> = historical_cycles_data.iter().cloned().collect();
    let (lifetime_mean, lifetime_std_deviation, _lifetime_max, _lifetime_min) = calculate_statistics(&historical_cycles_data_vec);

    println!("Lifetime Mean: {}, Lifetime Std Dev: {}", lifetime_mean, lifetime_std_deviation);
    }
     // Process and convert VecDeque to Vec for calculation
     let cycles_data_simple_vec: Vec<u64> = cycles_data_simple.into_iter().collect();
     let cycles_data_basic_vec: Vec<u64> = cycles_data_basic.into_iter().collect();
 
     let historical_cycles_data_vec: Vec<u64> = historical_cycles_data.into_iter().collect();
     let (lifetime_mean, lifetime_std_dev, lifetime_max, lifetime_min) = calculate_statistics(&historical_cycles_data_vec);
     let (mean_simple, std_dev_simple, max_simple, min_simple) = calculate_statistics(&cycles_data_simple_vec);
     let (mean_basic, std_dev_basic, max_basic, min_basic) = calculate_statistics(&cycles_data_basic_vec);
 
     // Combine simple and basic data or use them separately as needed
     let combined_cycles = [cycles_data_simple_vec, cycles_data_basic_vec].concat();
     append_cycle_data_to_file(DATA_FILE_PATH, &combined_cycles)?;
 
     // Finalize CycleDataStats struct
     let stats = CycleDataStats {
         cycle_counts: combined_cycles,
         mean_cycles: (mean_simple + mean_basic) / 2.0, // Example of combining data
         std_dev_cycles: (std_dev_simple + std_dev_basic) / 2.0, // Modify as needed
         max_cycle: std::cmp::max(max_simple, max_basic),
         min_cycle: std::cmp::min(min_simple, min_basic),
         label,
     };
 
     Ok(stats)
 }
