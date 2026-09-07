# Backward-compatible entry point for existing users. New installations should
# source shell/template-daily.sh as documented in README.md.
TEMPLATE_DAILY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -r "${TEMPLATE_DAILY_ROOT}/shell/template-daily.sh" ]; then
    source "${TEMPLATE_DAILY_ROOT}/shell/template-daily.sh"
else
    printf 'WARNING: template shell loader is missing; helpers were not loaded.\n' >&2
fi
unset TEMPLATE_DAILY_ROOT
