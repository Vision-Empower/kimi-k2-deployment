#!/usr/bin/env python3
"""
NUMA-aware optimization for Kimi-K2 CPU inference
Distributes MoE experts across NUMA nodes for optimal memory access
"""

import os
import subprocess
import psutil
import numpy as np
from typing import Dict, List, Tuple

class NUMAOptimizer:
    def __init__(self):
        self.numa_nodes = self._detect_numa_nodes()
        self.cpu_topology = self._get_cpu_topology()
        self.memory_info = self._get_memory_info()
        
    def _detect_numa_nodes(self) -> int:
        """Detect number of NUMA nodes"""
        try:
            result = subprocess.run(['numactl', '--hardware'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.startswith('available:'):
                    return int(line.split()[1])
        except:
            pass
        return 1
    
    def _get_cpu_topology(self) -> Dict[int, List[int]]:
        """Map NUMA nodes to CPU cores"""
        topology = {}
        try:
            for node in range(self.numa_nodes):
                result = subprocess.run(['numactl', '--cpubind', str(node), '--show'],
                                      capture_output=True, text=True)
                cpus = []
                for line in result.stdout.split('\n'):
                    if line.startswith('cpubind:'):
                        cpu_str = line.split(':')[1].strip()
                        # Parse CPU ranges like "0-47,96-143"
                        for cpu_range in cpu_str.split(','):
                            if '-' in cpu_range:
                                start, end = map(int, cpu_range.split('-'))
                                cpus.extend(range(start, end + 1))
                            else:
                                cpus.append(int(cpu_range))
                topology[node] = cpus
        except:
            # Fallback to even distribution
            total_cpus = psutil.cpu_count()
            cpus_per_node = total_cpus // self.numa_nodes
            for node in range(self.numa_nodes):
                start = node * cpus_per_node
                end = start + cpus_per_node
                topology[node] = list(range(start, end))
        return topology
    
    def _get_memory_info(self) -> Dict[int, int]:
        """Get memory size per NUMA node"""
        memory_info = {}
        try:
            for node in range(self.numa_nodes):
                result = subprocess.run(['numactl', '--hardware'],
                                      capture_output=True, text=True)
                # Parse memory info from numactl output
                for line in result.stdout.split('\n'):
                    if f'node {node} size:' in line:
                        # Extract memory size in MB
                        size_mb = int(line.split(':')[1].split()[0])
                        memory_info[node] = size_mb * 1024 * 1024  # Convert to bytes
        except:
            # Fallback to even distribution
            total_memory = psutil.virtual_memory().total
            memory_per_node = total_memory // self.numa_nodes
            for node in range(self.numa_nodes):
                memory_info[node] = memory_per_node
        return memory_info
    
    def optimize_expert_placement(self, num_experts: int) -> Dict[int, int]:
        """Distribute experts across NUMA nodes for optimal performance"""
        expert_to_node = {}
        
        # Distribute experts evenly across NUMA nodes
        experts_per_node = num_experts // self.numa_nodes
        remainder = num_experts % self.numa_nodes
        
        expert_id = 0
        for node in range(self.numa_nodes):
            node_experts = experts_per_node + (1 if node < remainder else 0)
            for _ in range(node_experts):
                expert_to_node[expert_id] = node
                expert_id += 1
                
        return expert_to_node
    
    def bind_thread_to_node(self, thread_id: int, numa_node: int):
        """Bind a thread to specific NUMA node CPUs"""
        cpus = self.cpu_topology[numa_node]
        # Round-robin assignment within node
        cpu_id = cpus[thread_id % len(cpus)]
        
        # Set CPU affinity
        try:
            os.sched_setaffinity(0, {cpu_id})
        except:
            print(f"Warning: Could not set CPU affinity for thread {thread_id}")
    
    def set_memory_policy(self, numa_node: int):
        """Set memory allocation policy for current process"""
        try:
            subprocess.run(['numactl', '--membind', str(numa_node), 
                          '--cpubind', str(numa_node)], check=True)
        except:
            print(f"Warning: Could not set NUMA memory policy")
    
    def get_optimization_summary(self) -> str:
        """Get summary of NUMA optimization settings"""
        summary = f"NUMA Optimization Summary:\n"
        summary += f"  NUMA Nodes: {self.numa_nodes}\n"
        summary += f"  CPU Topology:\n"
        for node, cpus in self.cpu_topology.items():
            summary += f"    Node {node}: CPUs {min(cpus)}-{max(cpus)} ({len(cpus)} cores)\n"
        summary += f"  Memory Distribution:\n"
        for node, memory in self.memory_info.items():
            summary += f"    Node {node}: {memory / (1024**3):.1f} GB\n"
        return summary

def setup_numa_optimized_inference():
    """Setup NUMA optimization for inference"""
    optimizer = NUMAOptimizer()
    
    # Print optimization summary
    print(optimizer.get_optimization_summary())
    
    # Set OpenMP environment variables for NUMA awareness
    os.environ['OMP_PROC_BIND'] = 'true'
    os.environ['OMP_PLACES'] = 'cores'
    os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
    
    # Intel MKL NUMA settings
    os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count())
    os.environ['MKL_DYNAMIC'] = 'false'
    
    # KMP settings for better NUMA performance
    os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '0'
    
    return optimizer

if __name__ == "__main__":
    # Test NUMA optimization
    optimizer = setup_numa_optimized_inference()
    
    # Example: Optimize placement for 128 experts
    expert_placement = optimizer.optimize_expert_placement(128)
    print(f"\nExpert placement across NUMA nodes:")
    for node in range(optimizer.numa_nodes):
        node_experts = [e for e, n in expert_placement.items() if n == node]
        print(f"  Node {node}: {len(node_experts)} experts (IDs: {node_experts[:5]}...)")