locals {
  bucket_name            = coalesce(var.bucket_name, "${var.project_id}-${var.name_prefix}")
  worker_service_account = "${var.name_prefix}-worker"
  worker_tag             = "${var.name_prefix}-worker"
}

resource "google_project_service" "compute" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_compute_network" "corpus" {
  project                 = var.project_id
  name                    = var.name_prefix
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.compute]
}

resource "google_compute_subnetwork" "workers" {
  project                  = var.project_id
  name                     = "${var.name_prefix}-workers"
  region                   = var.region
  network                  = google_compute_network.corpus.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
}

resource "google_compute_firewall" "worker_internal" {
  project = var.project_id
  name    = "${var.name_prefix}-internal"
  network = google_compute_network.corpus.name

  direction     = "INGRESS"
  source_ranges = [var.subnet_cidr]
  target_tags   = [local.worker_tag]

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }
}

resource "google_compute_firewall" "iap_ssh" {
  count = var.iap_ssh_enabled ? 1 : 0

  project = var.project_id
  name    = "${var.name_prefix}-iap-ssh"
  network = google_compute_network.corpus.name

  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"]
  target_tags   = [local.worker_tag]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "admin_ssh" {
  count = length(var.admin_ssh_source_ranges) == 0 ? 0 : 1

  project = var.project_id
  name    = "${var.name_prefix}-admin-ssh"
  network = google_compute_network.corpus.name

  direction     = "INGRESS"
  source_ranges = var.admin_ssh_source_ranges
  target_tags   = [local.worker_tag]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_service_account" "worker" {
  project      = var.project_id
  account_id   = local.worker_service_account
  display_name = "cppmega ephemeral corpus worker"

  depends_on = [google_project_service.compute]
}

resource "google_storage_bucket" "corpus" {
  project                     = var.project_id
  name                        = local.bucket_name
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  soft_delete_policy {
    retention_duration_seconds = var.soft_delete_retention_seconds
  }

  labels = {
    application = "cppmega"
    purpose     = "canonical-corpus-artifacts"
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.storage]
}

resource "google_storage_bucket_iam_member" "worker_object_viewer" {
  bucket = google_storage_bucket.corpus.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.worker.email}"

  condition {
    title       = "read_corpus_prefix"
    description = "Permit exact object reads only below the corpus prefix; listing is deliberately denied."
    expression  = "resource.name.startsWith('projects/_/buckets/${google_storage_bucket.corpus.name}/objects/${var.gcs_prefix}/')"
  }
}

resource "google_storage_bucket_iam_member" "worker_object_creator" {
  bucket = google_storage_bucket.corpus.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.worker.email}"

  condition {
    title       = "create_corpus_prefix"
    description = "Permit immutable object creation only below the corpus prefix."
    expression  = "resource.name.startsWith('projects/_/buckets/${google_storage_bucket.corpus.name}/objects/${var.gcs_prefix}/')"
  }
}
