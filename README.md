# UDF Benchmarking Example

Project showing how to perform some simple benchmarking of Python and Scala UDFs called from PySpark.

Note these benchmarks aren't rigorous, but should give an indication of performance in the prototyping stage of a project.

In particular the focus is on UDF performance on the executor, the overhead of network traffic is not taken into account.


## Approach

The project sets up a virtual machine (VM) using Vagrant.
The VM runs a standalone Spark cluster with 2 cores.

The test data distribution and volume should be roughly comparable to real data.

## Usage

From the terminal, create the virtual machine.

    vagrant up
    
Connect to the virtual machine.

    vagrant ssh
    
Run the benchmark

    ./run_benchmark.sh
    
Results will appear in `results`, logs in `logs`.

While the benchmark is running the Spark UI is available on `http://${IP}:4040`.
Where ${IP} is set by the ip argument on the config.vm.network line in the `Vagrantfile`.

##### Versions

This benchmarking example was created with the following versions of tools.

| Tool                          | Version       |
|-------------------------------|---------------|
| Vagrant                       | 2.1.5         |
| VirtualBox VM ubuntu/bionic64 | 20180913.0.0  |
| Spark                         | 2.3.1         |
| Miniconda                     | 4.5.1         |

Miniconda and pip are used to manage Python dependencies, the versions of dependencies aren't fixed. 

##### JVM Warm-Up

The JVM proactively optimises frequently run methods.
A common benchmarking strategy is to run the relevant methods before the benchmark.

This hasn't been done in this benchmark, 
so the first 10,000 or so calls of the profiled method may take longer than subsequent calls.
