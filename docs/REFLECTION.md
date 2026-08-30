# Reflection (300–500 words)

##  What was the most challenging part?

**The CPU-vs-GPU training constraint on limited hardware. As the CPU cores 
were limited to 2, the training took almost 20 hours (including a 30 mins
data download) to complete. Waiting for the training to complete, monitoring
overnight, was the most challenging part**


1. **The core challenge.** The obvious plan (train inside the Kubernetes Job 
   on the local kind cluster) ran into a hard reality: CPU training of ResNet-18
   on full CIFAR-10 took ~2 hours per epoch, versus ~1 minute per epoch on a 
   Colab T4 GPU , a difference of roughly two orders of magnitude.
  

2. **CPU bottleneck not Memory.** On a 8GB RAM system, During the in-cluster
   run, `docker stats` showed the container saturating its CPU allocation (215%,
   i.e. ~2 cores, at the spec's 2-core limit) while using only ~50% of its memory
   budget. This made it clear the workload was **CPU-bound, not memory-bound** 
   This was suprising as the expecatation was a OOM instead of CPU bottleneck.
   The training should have been started with 4 cores in hindsight, but started
   with the spec provided (CPU:2) and once halfway you cant change the number of 
   cores for the POD without killing it. Even then , it is not even comparable
   to what a GPU accomplished in terms of time. 

  The biggest learning from this exercise was about how effective deployment of
  training makes a world of difference. For real production systems, a CPU training 
  is almost impossible. Even though the mac (apple silicon) environment has integrated 
  GPU , because the Docker desktop runs a Linux VM , there is no other way , but to 
  use the CPU for training, when the training is local. This was in summary the biggest
  challenge and the biggest learnign from this exercise