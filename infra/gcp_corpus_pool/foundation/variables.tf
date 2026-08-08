variable "project_id" {
  description = "Google Cloud project that owns the corpus pool."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid Google Cloud project ID."
  }
}

variable "region" {
  description = "Region shared by the workers and the canonical Cloud Storage bucket."
  type        = string
  default     = "us-central1"
}

variable "name_prefix" {
  description = "Prefix for long-lived foundation resources."
  type        = string
  default     = "cppmega-corpus"

  validation {
    condition     = can(regex("^[a-z]([-a-z0-9]{0,21}[a-z0-9])?$", var.name_prefix))
    error_message = "name_prefix must be at most 23 lowercase letters, digits, and hyphens."
  }
}

variable "subnet_cidr" {
  description = "Private IPv4 range for corpus workers."
  type        = string
  default     = "10.42.0.0/24"

  validation {
    condition     = can(cidrhost(var.subnet_cidr, 1))
    error_message = "subnet_cidr must be a valid IPv4 CIDR."
  }
}

variable "bucket_name" {
  description = "Globally unique canonical artifact bucket. Null derives a name from project_id and name_prefix."
  type        = string
  default     = null
}

variable "gcs_prefix" {
  description = "Top-level object prefix reserved for corpus runs."
  type        = string
  default     = "runs"

  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,126}[a-zA-Z0-9]$", var.gcs_prefix)) && !strcontains(var.gcs_prefix, "//")
    error_message = "gcs_prefix must be a non-empty, slash-separated object prefix without a trailing slash."
  }
}

variable "soft_delete_retention_seconds" {
  description = "Cloud Storage soft-delete window. Set to zero to disable soft delete."
  type        = number
  default     = 604800

  validation {
    condition     = var.soft_delete_retention_seconds == 0 || (var.soft_delete_retention_seconds >= 604800 && var.soft_delete_retention_seconds <= 7776000)
    error_message = "soft-delete retention must be zero or between 7 and 90 days."
  }
}

variable "iap_ssh_enabled" {
  description = "Allow SSH from Identity-Aware Proxy TCP forwarding addresses."
  type        = bool
  default     = true
}

variable "admin_ssh_source_ranges" {
  description = "Optional additional trusted CIDRs allowed to SSH to workers. Empty by default."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for cidr in var.admin_ssh_source_ranges : can(cidrhost(cidr, 0))])
    error_message = "Every admin SSH source range must be valid CIDR notation."
  }
}
