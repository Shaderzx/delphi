import type { E2EConfig } from 'e2e';

export default {
  targets: [{ name: 'cli', platform: 'api' }],
  workers: 1,
  retries: 0,
  timeout: 60_000,
} satisfies E2EConfig;