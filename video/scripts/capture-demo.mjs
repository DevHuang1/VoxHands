// Capture live screenshots of the deployed VoxHands dashboard for the deck.
//
// Usage: node scripts/capture-demo.mjs [baseUrl]
//
// Uses the Chrome build that Remotion already downloaded, so no extra
// dependency is required. The refusal frame is produced by driving the server
// API (POST /api/command) rather than clicking in the page, because the
// headless CLI cannot script interaction.

import {spawnSync} from "node:child_process";
import {existsSync, mkdirSync} from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const videoRoot = path.resolve(here, "..");

const BASE = process.argv[2] ?? "https://voxhands.onrender.com";
const OUT_DIR = path.join(videoRoot, "public", "demo");
const REFUSAL_COMMAND = "Move the blue plate to the ground";

const findChrome = () => {
  const remotionDir = path.join(videoRoot, "node_modules", ".remotion");
  const candidates = [
    "chrome-headless-shell/mac-arm64/chrome-headless-shell-mac-arm64/chrome-headless-shell",
    "chrome-headless-shell/mac-x64/chrome-headless-shell-mac-x64/chrome-headless-shell",
    "chrome-headless-shell/linux64/chrome-headless-shell-linux64/chrome-headless-shell",
  ];
  for (const candidate of candidates) {
    const full = path.join(remotionDir, candidate);
    if (existsSync(full)) {
      return full;
    }
  }
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) {
    return process.env.CHROME_PATH;
  }
  throw new Error(`Could not find chrome-headless-shell under ${remotionDir}`);
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const post = async (route, body) => {
  const response = await fetch(`${BASE}${route}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: body === undefined ? "{}" : JSON.stringify(body),
  });
  return {status: response.status, payload: await response.json().catch(() => null)};
};

const waitForState = async (predicate, label, timeoutMs = 60000) => {
  const started = Date.now();
  let last = null;
  while (Date.now() - started < timeoutMs) {
    const state = await fetch(`${BASE}/api/state`).then((r) => r.json());
    last = state.status;
    if (predicate(state)) {
      return state;
    }
    await sleep(700);
  }
  console.warn(`[capture] timed out waiting for ${label}; last status was ${last}`);
  return null;
};

const shoot = (chrome, outFile, size = "1920,1080") => {
  const result = spawnSync(
    chrome,
    [
      "--hide-scrollbars",
      "--force-device-scale-factor=1",
      `--window-size=${size}`,
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--virtual-time-budget=12000",
      `--screenshot=${outFile}`,
      `${BASE}/`,
    ],
    {stdio: ["ignore", "ignore", "pipe"], encoding: "utf8"}
  );
  if (!existsSync(outFile)) {
    throw new Error(`Screenshot failed for ${outFile}: ${result.stderr?.slice(-400)}`);
  }
  console.log(`[capture] wrote ${path.relative(videoRoot, outFile)}`);
};

const main = async () => {
  const chrome = findChrome();
  console.log(`[capture] target ${BASE}`);
  console.log(`[capture] chrome ${path.relative(videoRoot, chrome)}`);
  mkdirSync(OUT_DIR, {recursive: true});

  await post("/api/reset");
  await waitForState((state) => state.status === "idle", "idle state");
  shoot(chrome, path.join(OUT_DIR, "dashboard.png"));
  shoot(chrome, path.join(OUT_DIR, "runtime.png"), "1920,2600");

  const refusal = await post("/api/command", {text: REFUSAL_COMMAND});
  console.log(`[capture] refusal request -> HTTP ${refusal.status}, status=${refusal.payload?.status}`);
  await waitForState((state) => state.status === "blocked", "blocked state");
  shoot(chrome, path.join(OUT_DIR, "refusal.png"));

  await post("/api/reset");
  await sleep(300);
  console.log("[capture] done");
};

main().catch((error) => {
  console.error(`[capture] failed: ${error.message}`);
  process.exit(1);
});
