import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor

class ProofValidator:
    """
    负责调用 Lean 对 .lean 文件进行编译 / 验证。

    Parameters
    ----------
    lean_cmd : List[str] | None
        执行 Lean 的命令行，默认 `["lake", "env", "lean"]`。
        如果你用 `lean --make` 或直接 `lean`, 可自行修改。
    timeout : int
        单文件验证超时时间（秒）。
    work_dir : str | Path | None
        若需要在特定目录下执行（例如项目根），可设置 cwd。
    """

    def __init__(
        self,
        lean_cmd: Optional[List[str]] = None,
        timeout: int = 100,
        work_dir: Optional[str | Path] = None,
    ):
        self.lean_cmd: List[str] = lean_cmd or ["lake", "env", "lean"]
        self.timeout = timeout
        self.work_dir = Path(work_dir) if work_dir else None

    # ------------------------------------------------------------------ #
    # 核心 API：验证单个 Lean 文件
    # ------------------------------------------------------------------ #
    def validate_file(self, filename: str | Path) -> Tuple[bool, str]:
        """
        调用 Lean 验证指定 .lean 文件。

        Returns
        -------
        (success, message):
            success : bool  —  True 表示验证通过
            message : str   —  Lean 的 stdout（成功）或 stderr（失败/超时）
        """
        filename = str(filename)

        try:
            result = subprocess.run(
                self.lean_cmd + [filename],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.work_dir,
            )
        except subprocess.TimeoutExpired:
            return False, "Lean 验证超时"

        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            # 失败时返回 stderr，方便调试
            return False, result.stdout.strip()

    # ------------------------------------------------------------------ #
    # （可选）直接验证字符串：先写临时文件再验证
    # ------------------------------------------------------------------ #
    def validate_code(
        self, code: str, tmp_dir: str | Path = "./tmp_proofs", stem: str = "proof"
    ) -> Tuple[bool, str]:
        """
        将 code 写入临时文件 <tmp_dir>/<stem>.lean 再调用 validate_file。
        """
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / f"{stem}.lean"
        tmp_file.write_text(code, encoding="utf-8")

        success, msg = self.validate_file(tmp_file)
        # 可按需删除/tmp_file 或保留看日志
        return success, msg
    
    def validate_code(
    self, code: str, tmp_dir: str | Path = "./tmp_proofs", stem: str = "proof"
) -> Tuple[bool, str]:
        """
        将 code 写入临时文件 <tmp_dir>/<stem>.lean，调用 validate_file 验证后自动删除临时文件。
        """
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / f"{stem}.lean"

        try:
            tmp_file.write_text(code, encoding="utf-8")
            success, msg = self.validate_file(tmp_file)
        finally:
            # 无论成功失败，最后都尝试删除临时文件
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete temp file {tmp_file}: {e}")

        return success, msg


    # ------------------------------------------------------------------ #
    # （可选）批量验证：返回通过 / 失败列表
    # ------------------------------------------------------------------ #
    def validate_batch(
        self, files: List[str | Path]
    ) -> Tuple[List[str], List[str]]:
        """
        批量验证 Lean 文件；返回 (passed_list, failed_list)。
        """
        passed, failed = [], []
        for f in files:
            ok, _ = self.validate_file(f)
            (passed if ok else failed).append(str(f))
        return passed, failed

    def validate_dir(
        self,num_workers: int = 1  
    ) -> List[Tuple[Path,bool,str]]:
        """
        对自身路径文件夹下的所有后缀为 .lean 的文件进行验证
        """
        base_dir = Path(self.work_dir)
        lean_files = list(base_dir.rglob("*.lean"))

        def validate_single(filepath: Path) -> Tuple[Path,bool,str]:
            success,running_message = self.validate_file(filepath)
            return (filepath,success,running_message)
        
        results: List[Tuple[Path,bool,str]] = []
        if num_workers == 1:
            for file in lean_files:
                results.append(validate_single(file))
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(validate_single,lean_files))
        return results
