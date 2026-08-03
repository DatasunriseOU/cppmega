output "bucket_name" {
  description = "Canonical Cloud Storage bucket retained after the worker pool is destroyed."
  value       = google_storage_bucket.corpus.name
}

output "gcs_root" {
  description = "Canonical object root for corpus runs."
  value       = "gs://${google_storage_bucket.corpus.name}/${var.gcs_prefix}"
}

output "network_name" {
  value = google_compute_network.corpus.name
}

output "subnetwork_name" {
  value = google_compute_subnetwork.workers.name
}

output "worker_service_account_email" {
  value = google_service_account.worker.email
}

output "worker_tag" {
  value = local.worker_tag
}
