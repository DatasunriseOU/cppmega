-- GH Archive extraction for the C/C++ corpus (data-cpp_all) PR ingest.
--
-- Pulls PullRequest / PullRequestReview / PullRequestReviewComment / IssueComment
-- events (the events that carry PR review discussion) for the resolved owner/repo
-- list, over a date range expressed as a wildcard table suffix.
--
-- This file is a TEMPLATE. scripts/pr_ingest/repo_list_from_tarball.py substitutes:
--   {repo_in_list}  -> 'owner/repo', 'owner2/repo2', ...   (the REAL resolved list)
--   {table_glob}    -> e.g. `githubarchive.month.202512`  (one month for dry-run)
--                      or   `githubarchive.month.20*`      (full history)
--
-- Cost note: a value filter on repo.name does NOT prune scanned bytes -- BigQuery
-- still reads the repo.name, type, payload, created_at, id columns across the whole
-- table glob. The dry-run reports the real bytesProcessed; cost = bytes/2^40 * $6.25.
SELECT
  type,
  repo.name        AS repo_name,
  actor.login      AS actor_login,
  created_at,
  id,
  payload
FROM `{table_glob}`
WHERE type IN (
        'PullRequestEvent',
        'PullRequestReviewEvent',
        'PullRequestReviewCommentEvent',
        'IssueCommentEvent'
      )
  AND repo.name IN ({repo_in_list})
