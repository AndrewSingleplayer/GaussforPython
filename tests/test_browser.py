"""The web AR page in a real browser (headless Chromium with a fake camera and simulated motion
sensors): starting AR, the ring on the floor, a tap placing the scene, no script errors.

Needs Playwright with Chromium and a built viewer (python3 web/build.py); skipped otherwise.
"""
import asyncio
import functools
import http.server
import os
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(os.path.dirname(HERE), "web", "dist")
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

try:
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None


def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@unittest.skipUnless(async_playwright and os.path.exists(os.path.join(DIST, "index.html")),
                     "needs Playwright and web/dist (python3 web/build.py)")
class ARPage(unittest.TestCase):
    def test_ar_starts_shows_the_ring_and_places_the_scene(self):
        server = serve(DIST)
        try:
            result = asyncio.run(self.run_page(f"http://127.0.0.1:{server.server_address[1]}"))
        finally:
            server.shutdown()
        self.assertEqual(result["errors"], [], result)
        self.assertTrue(result["ar_on"], result)
        self.assertIn("Tap to put it on the", result["ring_hint"], result)
        self.assertIsNotNone(result["placed"])
        self.assertIn("Walk around it", result["placed_hint"])
        self.assertNotIn("went wrong", result["status"])

    async def run_page(self, origin):
        args = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist",
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
        async with async_playwright() as p:
            # the full Chromium (not the stripped-down headless shell), in its new headless mode
            kw = {"executable_path": CHROME} if os.path.exists(CHROME) else {"channel": "chromium"}
            browser = await p.chromium.launch(args=args, **kw)
            ctx = await browser.new_context(viewport={"width": 393, "height": 852}, is_mobile=True, has_touch=True)
            await ctx.grant_permissions(["camera"], origin=origin)
            page = await ctx.new_page()
            errors, console = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            await page.goto(origin + "/index.html?still=1&debug#unicorn")
            for _ in range(120):
                d = await page.text_content("#drawn")
                if d and d != "–" and not d.startswith("0 "):
                    break
                await page.wait_for_timeout(500)
            await page.click("#ar")
            # the phone held 40 degrees down, readings at 60 Hz
            await page.evaluate("""setInterval(() => window.dispatchEvent(new DeviceOrientationEvent(
                'deviceorientation', {alpha: 0, beta: 50, gamma: 0})), 16)""")
            await page.wait_for_timeout(3000)
            hint = lambda: page.evaluate("document.getElementById('ar-hint').textContent")
            ring_hint = await hint()
            await page.mouse.click(196, 300)
            await page.wait_for_timeout(1500)
            result = {"errors": errors, "ring_hint": ring_hint, "placed": await page.evaluate("ar.placed"),
                      "placed_hint": await hint(), "status": await page.text_content("#status") or "",
                      "ar_on": await page.evaluate("ar.on"),
                      "note": await page.evaluate("document.getElementById('ar-note').hidden ? '' : "
                                                  "document.getElementById('ar-note').textContent.trim()"),
                      "readings": await page.evaluate("ar.motionEvents"), "console": console[-10:]}
            await browser.close()
            return result


if __name__ == "__main__":
    unittest.main()
