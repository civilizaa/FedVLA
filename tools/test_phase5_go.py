#!/usr/bin/env python3
"""phase5_go.py 的离线单测：验证「失败必须被拦住」，而不是「跑通就算好」。

针对的正是这一轮最贵的风险 —— 上云后脚本"看起来成功"但其实没推上去/没起跑，
于是守护误判「主控消失」提前关机，10+ 小时算力白烧（铁律 B/J/Q）。

🔴 **本文件必须全程离线**：用 `with patched(mock):` 把云端原语替换掉。
   曾经的 bug：`load_module` 在 import 后立刻 `finally` 还原 ⇒ 用例跑的是**真 relay**
   ⇒ 真的往云端入队了 11 条命令。现在有两道保险：
     ① `patched` 上下文管理器在用例**执行期间**保持 mock 生效；
     ② 每个用例结束断言 `mock.calls > 0` —— 没走到 mock 就是没生效，直接判失败。

跑法：cd tools && python test_phase5_go.py
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CASES = []
MOCK_KEYS = ("health", "req", "enqueue", "wait_results", "md5_local",
             "b64_put_cmd", "fix_remote")


def case(fn):
    CASES.append(fn)
    return fn


# ---------------------------------------------------------------- mock
class MockPF:
    """替换 push_files 的云端原语。calls 记录被调用的次数（用于断言 mock 生效）。"""

    def __init__(self):
        self.enqueued = []
        self.results = {}
        self.next_cid = 100
        self.handler = None
        self.calls = 0
        self.remotes = []

    def _bump(self):
        self.calls += 1

    def health(self):
        self._bump()
        return {"ok": True, "queued": 0}

    def req(self, path, data=None, timeout=30, retries=3):
        self._bump()
        return {"ok": True, "cleared": 0}

    def enqueue(self, cmd, timeout=300, retries=6):
        self._bump()
        self.next_cid += 1
        cid = self.next_cid
        self.enqueued.append((cid, cmd))
        if cmd.startswith("PUT "):
            self.remotes.append(cmd[4:])
        if self.handler:
            rc, out, err = self.handler(cmd)
            self.results[cid] = {"rc": rc, "out": out, "err": err}
        else:
            self.results[cid] = {"rc": 0, "out": "", "err": ""}
        return cid, "queued"

    def wait_results(self, cids, timeout=900):
        self._bump()
        return {c: self.results.get(c, {}) for c in cids}

    def md5_local(self, path):
        self._bump()
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def b64_put_cmd(self, text, remote):
        self._bump()
        return f"PUT {remote}"

    def fix_remote(self, remote):
        self._bump()
        return remote


@contextlib.contextmanager
def patched(mock):
    """把云端原语换成 mock，**在整个 with 块内保持生效**。

    🔴 曾经的坑：还原写在 import 的 finally 里 ⇒ 用例执行时已经换回真 relay。
    """
    import push_files as real_pf
    saved = {k: getattr(real_pf, k) for k in MOCK_KEYS}
    for k in MOCK_KEYS:
        setattr(real_pf, k, getattr(mock, k))
    # 重新载入 phase5_go，让它拿到被 patch 的模块引用
    sys.modules.pop("phase5_go", None)
    try:
        import phase5_go
        yield phase5_go
    finally:
        for k, v in saved.items():
            setattr(real_pf, k, v)
        sys.modules.pop("phase5_go", None)


# ---------------------------------------------------------------- 用例
@case
def test_push_detects_md5_mismatch():
    """md5 对不上 ⇒ step_push_impl 必须返回 False（不能"推过了就算"）。"""
    mock = MockPF()

    def handler(cmd):
        if cmd.startswith("md5sum"):
            return 0, "deadbeefdeadbeefdeadbeefdeadbeef  /root/HPT/federated/run_phase5.sh\n", ""
        return 0, "", ""

    mock.handler = handler
    with patched(mock) as m:
        orig_sleep = m.time.sleep
        m.time.sleep = lambda *_: None
        try:
            rc = m.step_push_impl()
        finally:
            m.time.sleep = orig_sleep
    assert mock.calls > 0, "mock 未生效（测试打到了真 relay！）"
    assert rc is False, f"md5 不一致却返回了 {rc}（应 False）"
    print("  ✅ md5 不一致被拦截")


@case
def test_push_ok_when_md5_matches():
    """md5 全对上 ⇒ 必须返回 True。"""
    mock = MockPF()

    def handler(cmd):
        if cmd.startswith("md5sum"):
            lines = [f"{mock.md5_local(os.path.join(mock.root, lp))}  {rp}"
                     for lp, rp in mock.file_map]
            return 0, "\n".join(lines) + "\n", ""
        return 0, "", ""

    mock.handler = handler
    with patched(mock) as m:
        mock.root = m.ROOT
        mock.file_map = m.FILES
        orig_sleep = m.time.sleep
        m.time.sleep = lambda *_: None
        try:
            rc = m.step_push_impl()
        finally:
            m.time.sleep = orig_sleep
    assert mock.calls > 0, "mock 未生效（测试打到了真 relay！）"
    assert rc is True, f"md5 全对却返回了 {rc}（应 True）"
    print("  ✅ md5 一致时放行")


@case
def test_test_step_blocks_on_failure():
    """单测不全绿 ⇒ step_test 必须返回 False（fail-loud，绝不开跑）。"""
    mock = MockPF()
    mock.handler = lambda cmd: (1, "Traceback ... 断言失败", "")
    with patched(mock) as m:
        rc = m.step_test()
    assert mock.calls > 0, "mock 未生效"
    assert rc is False, f"单测失败却返回了 {rc}（应 False）"
    print("  ✅ 单测失败被拦截")


@case
def test_test_step_passes_when_green():
    """单测全绿 ⇒ step_test 返回 True。"""
    mock = MockPF()
    mock.handler = lambda cmd: (0, "✅ 全部 9 个用例通过", "")
    with patched(mock) as m:
        rc = m.step_test()
    assert mock.calls > 0, "mock 未生效"
    assert rc is True, f"单测全绿却返回了 {rc}（应 True）"
    print("  ✅ 单测全绿时放行")


@case
def test_launch_blocks_when_no_worker():
    """45s 后看不到训练进程 ⇒ 绝不起守护（否则误触发提前关机）。"""
    mock = MockPF()

    def handler(cmd):
        if "ps -eo" in cmd:
            return 0, "(空)  ---轮次---\n", ""
        return 0, "", ""

    mock.handler = handler
    with patched(mock) as m:
        orig_sleep = m.time.sleep
        m.time.sleep = lambda *_: None
        try:
            rc = m.step_launch()
        finally:
            m.time.sleep = orig_sleep
    assert mock.calls > 0, "mock 未生效"
    assert rc is False, f"无 worker 却返回了 {rc}（应 False）"
    assert not any("phase5_autoshutdown" in c for _, c in mock.enqueued), \
        "训练没起来却仍去起了守护"
    print("  ✅ 无 worker 时拒绝起守护")


@case
def test_patch_step_requires_anchor():
    """train_test.py 里没有 apply_baseline_hook ⇒ step_patch 必须 False。"""
    mock = MockPF()

    def handler(cmd):
        if "--verify" in cmd:
            return 0, "[OK] 补丁已就位", ""
        if cmd.startswith("grep -c"):
            return 0, "0\n", ""
        return 0, "[OK] 补丁已应用", ""

    mock.handler = handler
    with patched(mock) as m:
        rc = m.step_patch()
    assert mock.calls > 0, "mock 未生效"
    assert rc is False, f"锚点缺失却返回了 {rc}（应 False）"
    print("  ✅ 锚点缺失被拦截")


@case
def test_push_idbase_is_increasing():
    """🔴 铁律 I：PUSH_IDBASE 必须 > 第 4 阶段的 1000000（同 idbase 会静默取到过期数据）。"""
    mock = MockPF()
    with patched(mock) as m:
        assert m.PUSH_IDBASE == 1010000, f"PUSH_IDBASE={m.PUSH_IDBASE}"
        assert m.PUSH_IDBASE > 1000000
    print("  ✅ idbase 递增")


def main() -> int:
    print("=" * 60)
    print("phase5_go.py 离线单测（全程离线，mock 生效有断言）")
    print("=" * 60)
    fail = 0
    for fn in CASES:
        print(f"[{fn.__name__}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  ❌ 断言失败：{e}")
            fail += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ 异常：{type(e).__name__}: {e}")
            fail += 1
    print("=" * 60)
    if fail:
        print(f"❌ {fail}/{len(CASES)} 个用例失败")
        return 1
    print(f"✅ 全部 {len(CASES)} 个用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
