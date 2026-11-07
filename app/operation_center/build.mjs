import { createRequire } from 'node:module';
import { mkdir, copyFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const app = path.dirname(fileURLToPath(import.meta.url));
const work = path.resolve(app, '../../.agent/operation_center');
const require = createRequire(path.join(work, 'node/package.json'));
const { build } = require('esbuild');
const output = path.join(work, 'ui');
await mkdir(output, { recursive: true });
await build({
  entryPoints: [path.join(app, 'ui/main.js')],
  outfile: path.join(output, 'app.js'),
  nodePaths: [path.join(work, 'node/node_modules')],
  bundle: true, format: 'esm', target: 'es2022', minify: true,
  legalComments: 'linked',
});
for (const file of ['index.html', 'style.css']) {
  await copyFile(path.join(app, 'ui', file), path.join(output, file));
}
