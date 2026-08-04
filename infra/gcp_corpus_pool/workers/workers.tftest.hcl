mock_provider "google" {}

run "default_four_worker_plan" {
  command = plan

  variables {
    project_id                = "natural-bison-491019-t9"
    run_id                    = "source-20260803-001"
    bootstrap_script_gcs_uri  = ""
    bootstrap_script_sha256   = ""
    bootstrap_bundle_sha256   = ""
    bootstrap_overlay_sha256  = ""
    bootstrap_manifest_sha256 = ""
  }

  assert {
    condition     = length(google_compute_instance.worker) == 4
    error_message = "The default pool must contain four workers."
  }

  assert {
    condition     = alltrue([for worker in google_compute_instance.worker : worker.machine_type == "n2-standard-16"])
    error_message = "Every default worker must be n2-standard-16."
  }

  assert {
    condition     = alltrue([for worker in google_compute_instance.worker : length(worker.scratch_disk) == 2])
    error_message = "Every default worker must have two Local SSD devices."
  }

  assert {
    condition     = alltrue([for worker in google_compute_instance.worker : worker.network_interface[0].nic_type == "GVNIC"])
    error_message = "Every worker must use gVNIC."
  }

  assert {
    condition     = length(google_compute_address.worker) == 4
    error_message = "Every default worker must have one reserved address."
  }
}

run "zero_workers_returns_pool" {
  command = plan

  variables {
    project_id                = "natural-bison-491019-t9"
    run_id                    = "source-20260803-001"
    worker_count              = 0
    bootstrap_script_gcs_uri  = ""
    bootstrap_script_sha256   = ""
    bootstrap_bundle_sha256   = ""
    bootstrap_overlay_sha256  = ""
    bootstrap_manifest_sha256 = ""
  }

  assert {
    condition     = length(google_compute_instance.worker) == 0 && length(google_compute_address.worker) == 0
    error_message = "worker_count zero must remove all disposable compute and addresses."
  }
}

run "content_addressed_v4_bootstrap" {
  command = plan

  variables {
    project_id                = "natural-bison-491019-t9"
    run_id                    = "source-pilot-20260803-002"
    bootstrap_script_gcs_uri  = "gs://natural-bison-491019-t9-cppmega-corpus/runs/source-pilot-20260803-002/bootstrap/cf62ebff18126915061ad2505f3f4773bb4e1d5702d78e50e51291fea3f79cf4.source-worker-runner"
    bootstrap_script_sha256   = "cf62ebff18126915061ad2505f3f4773bb4e1d5702d78e50e51291fea3f79cf4"
    bootstrap_bundle_sha256   = "cf312c1216ce521eedd5412b619d6de238f9baced1e9933b549faf366cd2445d"
    bootstrap_overlay_sha256  = "c075c268145782c6956d4ac8d13b26a73e6ae7aedc815debfb4056b1e464d323"
    bootstrap_manifest_sha256 = "8b928428520138ab90700640af3f2811bc988e3de883442eed938f28eb054ae2"
  }

  assert {
    condition     = strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "readonly BUNDLE_SHA256=\\\"$EXPECTED_BUNDLE_SHA256\\\"")
    error_message = "Startup must verify that the runner embeds the exact bundle digest."
  }

  assert {
    condition     = strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "cf62ebff18126915061ad2505f3f4773bb4e1d5702d78e50e51291fea3f79cf4.source-worker-runner")
    error_message = "Startup must download the content-addressed live v4 runner."
  }

  assert {
    condition     = strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "gcs_read_exact \"$BOOTSTRAP_URI\"") && !strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "gcloud storage cp \"$BOOTSTRAP_URI\"")
    error_message = "Startup must use the v4-proven exact-object download path."
  }

  assert {
    condition = (
      strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "systemctl enable --now cppmega-source-worker.service") &&
      strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "Restart=on-failure") &&
      strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "StartLimitIntervalSec=0") &&
      !strcontains(google_compute_instance.worker["cppmega-corpus-00"].metadata_startup_script, "runuser -u cppmega")
    )
    error_message = "Content-addressed workers must run under an unbounded on-failure systemd supervisor."
  }
}

run "bounded_two_slot_profile" {
  command = plan

  variables {
    project_id                = "natural-bison-491019-t9"
    run_id                    = "source-two-slot-test"
    worker_count              = 4
    machine_type              = "n2-standard-16"
    slots_per_worker          = 2
    parse_workers_per_slot    = 6
    memory_limit_gb_per_slot  = 24
    cpu_budget_vcpus          = 16
    memory_budget_gb          = 56
    bootstrap_script_gcs_uri  = ""
    bootstrap_script_sha256   = ""
    bootstrap_bundle_sha256   = ""
    bootstrap_overlay_sha256  = ""
    bootstrap_manifest_sha256 = ""
  }

  assert {
    condition     = length(google_compute_instance.worker) == 4 && length(google_compute_address.worker) == 4
    error_message = "Multi-slot mode must preserve the physical VM and address count."
  }

  assert {
    condition     = alltrue([for worker in google_compute_instance.worker : strcontains(worker.metadata_startup_script, "SLOTS_PER_WORKER=2") && strcontains(worker.metadata_startup_script, "PARSE_WORKERS_PER_SLOT=6") && strcontains(worker.metadata_startup_script, "MEMORY_LIMIT_GB_PER_SLOT=24")])
    error_message = "Startup must carry the bounded two-slot resource profile."
  }

  assert {
    condition     = alltrue([for worker in google_compute_instance.worker : worker.metadata["cppmega-slots-per-worker"] == "2"])
    error_message = "Worker metadata must expose the physical-to-logical slot topology."
  }
}
