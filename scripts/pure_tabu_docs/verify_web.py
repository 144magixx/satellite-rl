"""Open presentation pages in Edge and check the paired-results UI."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs/Claude维护/导师汇报").glob("*/*.html"))
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
VISUALS = ROOT / "outputs/experiments/s2_pure_tabu_20261021/web_visual"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=str(EDGE))
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for path in PAGES:
            errors.clear()
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_timeout(350)
            if path.name.startswith(("04_", "18_", "19_近期")):
                rows = page.locator("#tabu-initialization-comparison tbody tr")
                assert rows.count() == 16, (path, rows.count())
                assert page.locator('[data-method="PTS320"]').inner_text().find("91.1%") >= 0
                for link in page.locator("#tabu-initialization-comparison a").all():
                    assert link.evaluate("a => new URL(a.href).protocol") == "file:"
                    target = Path(link.evaluate("a => decodeURIComponent(new URL(a.href).pathname.slice(1))"))
                    assert target.exists(), (path, target)
                VISUALS.mkdir(parents=True, exist_ok=True)
                for width, label in ((1280, "desktop"), (390, "mobile")):
                    page.set_viewport_size({"width": width, "height": 800})
                    section = page.locator("#tabu-initialization-comparison")
                    section.screenshot(path=str(VISUALS / f"{path.parent.name}_{label}.png"))
                    assert section.evaluate("el => el.scrollWidth <= el.clientWidth + 1"), (path, width)
                    if width == 390:
                        scroller = section.locator(".tabu-sync-scroll").first
                        assert scroller.evaluate("el => el.scrollWidth > el.clientWidth"), path
                    page.set_viewport_size({"width": 1280, "height": 800})
            if path.name.startswith("09_"):
                page.goto(path.as_uri() + "#tabu", wait_until="load")
                page.wait_for_timeout(500)
                assert page.locator("#tabuFrame").get_attribute("src").startswith("07_")
                assert page.frame_locator("#tabuFrame").locator("#headline").count() == 1
            assert not errors, (path, errors)
            print(path.relative_to(ROOT), "OK")
        browser.close()


if __name__ == "__main__":
    main()
