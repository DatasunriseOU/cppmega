locals {
  bucket_name            = coalesce(var.bucket_name, "${var.project_id}-${var.name_prefix}")
  network_name           = var.name_prefix
  subnetwork_name        = "${var.name_prefix}-workers"
  worker_service_account = "${var.name_prefix}-worker"
  worker_tag             = "${var.name_prefix}-worker"
  run_root               = "gs://${local.bucket_name}/${var.gcs_prefix}/${var.run_id}"
  worker_names = {
    for index in range(var.worker_count) : format("%s-%02d", var.name_prefix, index) => index
  }
}

data "google_compute_network" "corpus" {
  project = var.project_id
  name    = local.network_name
}

data "google_compute_subnetwork" "workers" {
  project = var.project_id
  name    = local.subnetwork_name
  region  = var.region
}

data "google_service_account" "worker" {
  project    = var.project_id
  account_id = local.worker_service_account
}

data "google_storage_bucket" "corpus" {
  name = local.bucket_name
}

data "google_compute_image" "worker" {
  project = var.image_project
  family  = var.image_family
}

resource "google_compute_resource_policy" "compact" {
  count = var.compact_placement && var.worker_count > 0 ? 1 : 0

  project = var.project_id
  name    = "${var.name_prefix}-${var.run_id}-compact"
  region  = var.region

  group_placement_policy {
    collocation = "COLLOCATED"
  }

  # GCP does not support renaming a placement policy. Keep the pool-level
  # policy identity stable while run-scoped workers are replaced.
  lifecycle {
    ignore_changes = [name]
  }
}

resource "google_compute_address" "worker" {
  for_each = local.worker_names

  project      = var.project_id
  name         = "${each.key}-${var.run_id}"
  region       = var.region
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
}

resource "google_compute_instance" "worker" {
  for_each = local.worker_names

  project                   = var.project_id
  name                      = "${each.key}-${var.run_id}"
  zone                      = var.zone
  machine_type              = var.machine_type
  allow_stopping_for_update = true
  can_ip_forward            = false
  deletion_protection       = false
  enable_display            = false
  resource_policies         = var.compact_placement ? [google_compute_resource_policy.compact[0].self_link] : []
  tags                      = [local.worker_tag]

  labels = merge(
    {
      application = "cppmega"
      lifecycle   = "ephemeral"
      run-id      = var.run_id
    },
    var.labels,
  )

  boot_disk {
    auto_delete = true

    initialize_params {
      image = data.google_compute_image.worker.self_link
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
  }

  dynamic "scratch_disk" {
    for_each = range(var.local_ssd_count)
    content {
      interface = "NVME"
    }
  }

  network_interface {
    network    = data.google_compute_network.corpus.self_link
    subnetwork = data.google_compute_subnetwork.workers.self_link
    nic_type   = "GVNIC"

    access_config {
      nat_ip       = google_compute_address.worker[each.key].address
      network_tier = "PREMIUM"
    }
  }

  scheduling {
    automatic_restart           = var.use_spot ? false : true
    on_host_maintenance         = var.use_spot ? "TERMINATE" : "MIGRATE"
    preemptible                 = var.use_spot
    provisioning_model          = var.use_spot ? "SPOT" : "STANDARD"
    instance_termination_action = var.use_spot ? "DELETE" : null
  }

  service_account {
    email  = data.google_service_account.worker.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  metadata = {
    block-project-ssh-keys   = "TRUE"
    enable-oslogin           = "TRUE"
    serial-port-enable       = "TRUE"
    cppmega-run-root         = local.run_root
    cppmega-worker-index     = tostring(each.value)
    cppmega-worker-count     = tostring(var.worker_count)
    cppmega-slots-per-worker = tostring(var.slots_per_worker)
  }

  metadata_startup_script = templatefile("${path.module}/startup.sh.tftpl", {
    bootstrap_bundle_sha256   = var.bootstrap_bundle_sha256
    bootstrap_manifest_sha256 = var.bootstrap_manifest_sha256
    bootstrap_overlay_sha256  = var.bootstrap_overlay_sha256
    bootstrap_script_gcs_uri  = var.bootstrap_script_gcs_uri
    bootstrap_script_sha256   = var.bootstrap_script_sha256
    runner_role               = var.runner_role
    bucket_name               = local.bucket_name
    gcs_prefix                = var.gcs_prefix
    local_ssd_count           = var.local_ssd_count
    run_id                    = var.run_id
    worker_count              = var.worker_count
    worker_index              = each.value
    worker_name               = "${each.key}-${var.run_id}"
    slots_per_worker          = var.slots_per_worker
    parse_workers_per_slot    = var.parse_workers_per_slot
    memory_limit_gb_per_slot  = var.memory_limit_gb_per_slot
    cpu_budget_vcpus          = var.cpu_budget_vcpus
    memory_budget_gb          = var.memory_budget_gb
  })

  lifecycle {
    precondition {
      condition = alltrue([
        for value in [
          var.bootstrap_script_gcs_uri,
          var.bootstrap_script_sha256,
          var.bootstrap_bundle_sha256,
          var.bootstrap_overlay_sha256,
          var.bootstrap_manifest_sha256,
        ] : value == ""
        ]) || alltrue([
        for value in [
          var.bootstrap_script_gcs_uri,
          var.bootstrap_script_sha256,
          var.bootstrap_bundle_sha256,
          var.bootstrap_overlay_sha256,
          var.bootstrap_manifest_sha256,
        ] : value != ""
      ])
      error_message = "runner URI and runner/bundle/overlay/manifest SHA-256 values must either all be set or all be blank."
    }

    precondition {
      condition = var.bootstrap_script_gcs_uri == "" || (
        startswith(var.bootstrap_script_gcs_uri, "${local.run_root}/bootstrap/") &&
        endswith(var.bootstrap_script_gcs_uri, "${var.bootstrap_script_sha256}.${var.runner_role}-worker-runner")
      )
      error_message = "bootstrap_script_gcs_uri must be the content-addressed <run_root>/bootstrap/<sha256>.<runner_role>-worker-runner object."
    }

    precondition {
      condition     = startswith(var.zone, "${var.region}-")
      error_message = "zone must belong to region so workers and the canonical bucket remain colocated."
    }

    precondition {
      condition     = !var.compact_placement || var.worker_count <= 22
      error_message = "Google compact placement policies support at most 22 instances; disable compact_placement for a larger pool."
    }

    precondition {
      condition     = var.slots_per_worker * var.parse_workers_per_slot <= var.cpu_budget_vcpus
      error_message = "aggregate parser workers per VM exceed cpu_budget_vcpus."
    }

    precondition {
      condition     = var.slots_per_worker * var.memory_limit_gb_per_slot <= var.memory_budget_gb
      error_message = "aggregate slot memory limits per VM exceed memory_budget_gb."
    }
  }
}
