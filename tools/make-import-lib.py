"""从 DLL 生成 MSVC 导入库（.lib），用于链接 OBS 发布包里的 obs.dll。
需要在 MSVC 开发者环境里运行（要用 dumpbin 和 lib）。

用法：python tools/make-import-lib.py <dll 路径> <输出 .lib 路径>
"""
import os
import re
import subprocess
import sys


def main():
    dll, out_lib = sys.argv[1], sys.argv[2]
    text = subprocess.run(["dumpbin", "/nologo", "/exports", dll], capture_output=True, text=True,
                          check=True).stdout
    # 导出表的行：序号  提示  RVA  名称
    names = re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)", text, re.M)
    if not names:
        sys.exit("没有从 dumpbin 输出里解析到导出函数")
    base = os.path.splitext(os.path.basename(dll))[0]
    def_path = os.path.splitext(out_lib)[0] + ".def"
    with open(def_path, "w") as f:
        f.write(f"LIBRARY {base}\nEXPORTS\n")
        for n in names:
            f.write(f"    {n}\n")
    subprocess.run(["lib", "/nologo", f"/def:{def_path}", f"/out:{out_lib}", "/machine:x64"], check=True)
    print(f"{out_lib}：{len(names)} 个导出")


if __name__ == "__main__":
    main()
