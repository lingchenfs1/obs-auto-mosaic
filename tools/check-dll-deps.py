"""检查一个目录里所有 DLL 的导入依赖是否齐全（Windows，需要 MSVC 开发者环境里的 dumpbin）。
每个依赖必须满足其一：在同一目录里；在 C:\\Windows\\System32 里；是 api-ms-win-* / ext-ms-* 系统转发 DLL；
或在允许列表里（显卡驱动提供的 DLL，构建机器上没有，但用户装了驱动就有）。

用法：python tools/check-dll-deps.py <目录> [允许缺失的 DLL ...]
有缺失时打印并以非 0 退出。
"""
import os
import re
import subprocess
import sys


def dependents(dll):
    out = subprocess.run(["dumpbin", "/nologo", "/dependents", dll], capture_output=True, text=True,
                         check=True).stdout
    # 「Image has the following dependencies」和「delay load dependencies」两段里缩进的 *.dll 行
    return {m.lower() for m in re.findall(r"^\s+(\S+\.dll)\s*$", out, re.M | re.I)}


def main():
    folder = sys.argv[1]
    allowed = {a.lower() for a in sys.argv[2:]}
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    local = {f.lower() for f in os.listdir(folder) if f.lower().endswith(".dll")}
    missing = {}
    total = 0
    for f in sorted(local):
        for dep in dependents(os.path.join(folder, f)):
            total += 1
            if dep in local or dep in allowed or dep.startswith(("api-ms-win-", "ext-ms-")):
                continue
            if os.path.exists(os.path.join(system32, dep)):
                continue
            missing.setdefault(dep, []).append(f)
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f)))
    print(f"{folder}：{len(local)} 个 DLL，共 {size / 1e6:.0f} MB，检查了 {total} 条依赖")
    for dep, users in sorted(missing.items()):
        print(f"  缺少 {dep}（被 {', '.join(users)} 依赖）")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
