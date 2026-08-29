# mlops-pytorch-pipeline

End-to-end deployment of a PyTorch image classifier (CIFAR-10) through
Docker and Kubernetes, built for the MLOps & Infrastructure course.

## Status
Scaffolding in place. Implementation in progress across feature branches.

## Structure
- `src/` — model, dataset, training, serving code
- `docker/` — multi-stage training + slim serving images
- `k8s/` — Kubernetes manifests (Job, Deployment, Service, ConfigMap, HPA)
- `configs/` — training hyperparameters
- `requirements/` — pinned dependencies (train / serve split)
- `tests/` — unit tests

Full setup instructions and architecture diagram to follow.
