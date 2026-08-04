output "run_root" {
  description = "Canonical Cloud Storage root for this logical run."
  value       = local.run_root
}

output "workers" {
  description = "Deterministic map of physical VM identity, logical slot topology, and addresses."
  value = {
    for name, instance in google_compute_instance.worker : name => {
      instance_name        = instance.name
      shard_index          = local.worker_names[name]
      shard_count          = var.worker_count
      slots_per_worker     = var.slots_per_worker
      logical_worker_count = var.worker_count * var.slots_per_worker
      internal_ip          = instance.network_interface[0].network_ip
      external_ip          = google_compute_address.worker[name].address
      zone                 = instance.zone
      stage_gib            = var.local_ssd_count * 375
    }
  }
}

output "iap_ssh_commands" {
  description = "IAP SSH commands; OS Login IAM is still required for the caller."
  value = {
    for name, instance in google_compute_instance.worker : name =>
    "gcloud compute ssh ${instance.name} --project=${var.project_id} --zone=${var.zone} --tunnel-through-iap"
  }
}
