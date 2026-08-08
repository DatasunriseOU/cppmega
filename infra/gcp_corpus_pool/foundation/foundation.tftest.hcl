mock_provider "google" {}

run "default_foundation_plan" {
  command = plan

  variables {
    project_id = "natural-bison-491019-t9"
  }

  assert {
    condition     = google_storage_bucket.corpus.force_destroy == false
    error_message = "The canonical bucket must never use force_destroy."
  }

  assert {
    condition     = google_storage_bucket.corpus.uniform_bucket_level_access
    error_message = "The canonical bucket must use uniform bucket-level access."
  }

  assert {
    condition     = google_compute_subnetwork.workers.private_ip_google_access
    error_message = "Workers require private Google API access."
  }

  assert {
    condition     = google_compute_firewall.iap_ssh[0].source_ranges == toset(["35.235.240.0/20"])
    error_message = "IAP SSH must be restricted to Google's documented TCP forwarding range."
  }

  assert {
    condition     = google_storage_bucket_iam_member.worker_object_creator.condition[0].expression == "resource.name.startsWith('projects/_/buckets/natural-bison-491019-t9-cppmega-corpus/objects/runs/')"
    error_message = "Workers must only create objects below the configured corpus prefix."
  }
}
