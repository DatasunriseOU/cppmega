-- Tier-2 PR ingest: raw GH Archive events plus projected PR discussion fields.
--
-- The raw columns (type, actor_login, id, payload) preserve the root loader
-- contract. The projected columns are consumed directly by pr_store.py and
-- avoid reparsing payload when the query is used as the production path.
--
-- scripts/pr_ingest/gharchive_run.sh substitutes {table_glob} and
-- {repo_in_list}. Only GitHub owner/repo keys are queried; non-GitHub forge
-- identities remain in repo_list.json for source provenance but do not enter
-- this GitHub-only stream.

SELECT
  type,
  type AS event_type,
  repo.name AS repo_name,
  actor.login AS actor_login,
  created_at,
  id,
  payload,

  COALESCE(
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.number'),
    JSON_EXTRACT_SCALAR(payload, '$.issue.number')
  ) AS pr_number,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.merge_commit_sha')
    AS merge_commit_sha,
  JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS pr_merged,

  JSON_EXTRACT_SCALAR(payload, '$.pull_request.title') AS pr_title,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.body') AS pr_body,
  JSON_EXTRACT_SCALAR(payload, '$.issue.pull_request.url')
    AS issue_is_pr_url,
  JSON_EXTRACT_SCALAR(payload, '$.issue.title') AS issue_title,
  JSON_EXTRACT_SCALAR(payload, '$.issue.body') AS issue_body,

  JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS comment_body,
  JSON_EXTRACT_SCALAR(payload, '$.comment.user.login') AS comment_user,
  JSON_EXTRACT_SCALAR(payload, '$.comment.path') AS comment_path,
  JSON_EXTRACT_SCALAR(payload, '$.review.body') AS review_body,
  JSON_EXTRACT_SCALAR(payload, '$.review.state') AS review_state,
  JSON_EXTRACT_SCALAR(payload, '$.review.user.login') AS review_user
FROM `{table_glob}`
WHERE type IN (
  'PullRequestEvent',
  'PullRequestReviewEvent',
  'PullRequestReviewCommentEvent',
  'IssueCommentEvent'
)
  AND repo.name IN ({repo_in_list})
  AND (
    type != 'IssueCommentEvent'
    OR JSON_EXTRACT_SCALAR(payload, '$.issue.pull_request.url') IS NOT NULL
  )
ORDER BY repo_name, pr_number, created_at
