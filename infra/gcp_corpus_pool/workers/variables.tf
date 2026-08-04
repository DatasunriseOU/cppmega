variable "project_id" {
  description = "Google Cloud project containing the foundation resources."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid Google Cloud project ID."
  }
}

variable "region" {
  description = "Region containing the subnet, static addresses, and canonical bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Single worker zone. Keep it in region and near the regional bucket."
  type        = string
  default     = "us-central1-a"
}

variable "name_prefix" {
  description = "Prefix shared with the foundation stack."
  type        = string
  default     = "cppmega-corpus"

  validation {
    condition     = can(regex("^[a-z]([-a-z0-9]{0,21}[a-z0-9])?$", var.name_prefix))
    error_message = "name_prefix must be at most 23 lowercase letters, digits, and hyphens."
  }
}

variable "bucket_name" {
  description = "Existing canonical bucket from the foundation stack. Null derives the foundation default."
  type        = string
  default     = null
}

variable "gcs_prefix" {
  description = "Top-level object prefix shared with the foundation stack."
  type        = string
  default     = "runs"

  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,126}[a-zA-Z0-9]$", var.gcs_prefix)) && !strcontains(var.gcs_prefix, "//")
    error_message = "gcs_prefix must be a non-empty, slash-separated object prefix without a trailing slash."
  }
}

variable "run_id" {
  description = "Immutable logical run identifier used in every GCS object path and worker receipt."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]{0,26}[a-z0-9])?$", var.run_id))
    error_message = "run_id must be 1-28 lowercase letters, digits, or hyphens."
  }
}

variable "worker_count" {
  description = "Number of independent corpus workers. Set to zero to return the whole compute pool."
  type        = number
  default     = 4

  validation {
    condition     = var.worker_count >= 0 && var.worker_count <= 32 && floor(var.worker_count) == var.worker_count
    error_message = "worker_count must be an integer between 0 and 32."
  }
}

variable "slots_per_worker" {
  description = "Independent logical source-worker slots launched by each VM. One preserves the original smoke behavior; two is the bounded production profile."
  type        = number
  default     = 1

  validation {
    condition     = var.slots_per_worker >= 1 && var.slots_per_worker <= 2 && floor(var.slots_per_worker) == var.slots_per_worker
    error_message = "slots_per_worker must be either 1 or 2."
  }
}

variable "parse_workers_per_slot" {
  description = "Bounded clang indexer parser workers per logical slot."
  type        = number
  default     = 8

  validation {
    condition     = var.parse_workers_per_slot >= 1 && var.parse_workers_per_slot <= 16 && floor(var.parse_workers_per_slot) == var.parse_workers_per_slot
    error_message = "parse_workers_per_slot must be an integer between 1 and 16."
  }
}

variable "memory_limit_gb_per_slot" {
  description = "Indexer memory limit for each logical slot in GiB."
  type        = number
  default     = 48

  validation {
    condition     = var.memory_limit_gb_per_slot > 0 && var.memory_limit_gb_per_slot <= 64
    error_message = "memory_limit_gb_per_slot must be greater than zero and at most 64 GiB."
  }
}

variable "cpu_budget_vcpus" {
  description = "Aggregate parser-worker CPU budget per VM; the runner refuses to exceed visible host capacity."
  type        = number
  default     = 16

  validation {
    condition     = var.cpu_budget_vcpus >= 1 && var.cpu_budget_vcpus <= 128 && floor(var.cpu_budget_vcpus) == var.cpu_budget_vcpus
    error_message = "cpu_budget_vcpus must be an integer between 1 and 128."
  }
}

variable "memory_budget_gb" {
  description = "Aggregate slot memory budget per VM in GiB, leaving room for the OS and GCS transport."
  type        = number
  default     = 56

  validation {
    condition     = var.memory_budget_gb > 0 && var.memory_budget_gb <= 512
    error_message = "memory_budget_gb must be greater than zero and at most 512 GiB."
  }
}

variable "machine_type" {
  description = "Worker machine type. n2-standard-16 is 16 vCPU, 64 GB, and up to 32 Gbps default egress."
  type        = string
  default     = "n2-standard-16"
}

variable "local_ssd_count" {
  description = "Number of 375 GiB NVMe Local SSD devices per worker, striped as RAID0."
  type        = number
  default     = 2

  validation {
    condition     = contains([2, 4, 8, 16, 24], var.local_ssd_count)
    error_message = "N2 workers with 16 vCPUs support 2, 4, 8, 16, or 24 Local SSD devices."
  }
}

variable "boot_disk_size_gb" {
  description = "Auto-deleted persistent boot disk size. Corpus data must not be staged here."
  type        = number
  default     = 50

  validation {
    condition     = var.boot_disk_size_gb >= 20 && var.boot_disk_size_gb <= 200
    error_message = "boot_disk_size_gb must be between 20 and 200 GB."
  }
}

variable "image_project" {
  description = "Public image project for the boot image."
  type        = string
  default     = "debian-cloud"
}

variable "image_family" {
  description = "Boot image family with gVNIC support."
  type        = string
  default     = "debian-12"
}

variable "use_spot" {
  description = "Use interruptible Spot VMs. Safe only when every shard checkpoints to GCS."
  type        = bool
  default     = false
}

variable "compact_placement" {
  description = "Request compact placement for lower worker-to-worker latency. Disable if capacity is scarce."
  type        = bool
  default     = true
}

variable "runner_role" {
  description = "Runner/service role. Source remains the default; cloud-lane uses an independent state and runner suffix."
  type        = string
  default     = "source"

  validation {
    condition     = contains(["source", "cloud-lane"], var.runner_role)
    error_message = "runner_role must be either source or cloud-lane."
  }
}

variable "bootstrap_script_gcs_uri" {
  description = "Optional immutable content-addressed gs:// URI for the worker runner. Blank provisions ready workers without starting a job."
  type        = string
  default     = ""

  validation {
    condition     = var.bootstrap_script_gcs_uri == "" || can(regex("^gs://[a-z0-9][a-z0-9._-]+/.+", var.bootstrap_script_gcs_uri))
    error_message = "bootstrap_script_gcs_uri must be blank or an object URI beginning with gs://."
  }
}

variable "bootstrap_script_sha256" {
  description = "Required SHA-256 for bootstrap_script_gcs_uri. Prevents executing a mutable or corrupt runner."
  type        = string
  default     = ""

  validation {
    condition     = var.bootstrap_script_sha256 == "" || can(regex("^[0-9a-f]{64}$", var.bootstrap_script_sha256))
    error_message = "bootstrap_script_sha256 must be blank or 64 lowercase hexadecimal characters."
  }
}

variable "bootstrap_bundle_sha256" {
  description = "Optional SHA-256 of the cppmega git bundle pinned inside the runner. Required with a runner."
  type        = string
  default     = ""

  validation {
    condition     = var.bootstrap_bundle_sha256 == "" || can(regex("^[0-9a-f]{64}$", var.bootstrap_bundle_sha256))
    error_message = "bootstrap_bundle_sha256 must be blank or 64 lowercase hexadecimal characters."
  }
}

variable "bootstrap_overlay_sha256" {
  description = "Optional SHA-256 of the distributed-data-prep overlay pinned inside the runner. Required with a runner."
  type        = string
  default     = ""

  validation {
    condition     = var.bootstrap_overlay_sha256 == "" || can(regex("^[0-9a-f]{64}$", var.bootstrap_overlay_sha256))
    error_message = "bootstrap_overlay_sha256 must be blank or 64 lowercase hexadecimal characters."
  }
}

variable "bootstrap_manifest_sha256" {
  description = "Optional raw-file SHA-256 of the distributed source manifest pinned inside the runner. Required with a runner."
  type        = string
  default     = ""

  validation {
    condition     = var.bootstrap_manifest_sha256 == "" || can(regex("^[0-9a-f]{64}$", var.bootstrap_manifest_sha256))
    error_message = "bootstrap_manifest_sha256 must be blank or 64 lowercase hexadecimal characters."
  }
}

variable "labels" {
  description = "Additional labels applied to every ephemeral worker."
  type        = map(string)
  default     = {}
}
