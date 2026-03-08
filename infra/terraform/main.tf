# ============================================================================
# Terraform — GCP Cloud Run deployment for BellwetherML API
#
# What this provisions:
#   - Artifact Registry repo (stores your Docker image)
#   - Cloud Run service (auto-scales, serverless, HTTPS out of the box)
#   - IAM policy (public read for the prediction endpoint)
#
# Usage:
#   cd infra/terraform
#   terraform init
#   terraform plan -var="project_id=YOUR_GCP_PROJECT"
#   terraform apply -var="project_id=YOUR_GCP_PROJECT"
#
# Pre-requisites:
#   gcloud auth application-default login
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
# ============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-west1"   # Mountain View / Bay Area — low latency
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "model_name" {
  description = "ML model backend to use (random_forest or xgboost)"
  type        = string
  default     = "random_forest"
}

# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  service_name = "bellwether-ml-api"
  image_url    = "${var.region}-docker.pkg.dev/${var.project_id}/bellwether/${local.service_name}:${var.image_tag}"
}

# --------------------------------------------------------------------------- #
# Artifact Registry
# --------------------------------------------------------------------------- #

resource "google_artifact_registry_repository" "bellwether" {
  location      = var.region
  repository_id = "bellwether"
  format        = "DOCKER"
  description   = "BellwetherML Docker images"
}

# --------------------------------------------------------------------------- #
# Cloud Run Service
# --------------------------------------------------------------------------- #

resource "google_cloud_run_v2_service" "api" {
  name     = local.service_name
  location = var.region

  template {
    scaling {
      min_instance_count = 0    # scale to zero when idle
      max_instance_count = 10   # scale up under load
    }

    containers {
      image = local.image_url

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true   # only bill when processing requests
      }

      env {
        name  = "MODEL_NAME"
        value = var.model_name
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      ports {
        container_port = 8000
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 2
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_artifact_registry_repository.bellwether]
}

# --------------------------------------------------------------------------- #
# IAM — Allow unauthenticated access to /predict and /health
# --------------------------------------------------------------------------- #

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

output "api_url" {
  description = "Public URL of the deployed BellwetherML API"
  value       = google_cloud_run_v2_service.api.uri
}

output "image_push_command" {
  description = "Command to build and push Docker image"
  value       = <<-EOT
    docker build -t ${local.image_url} -f ../Dockerfile ../..
    docker push ${local.image_url}
  EOT
}
