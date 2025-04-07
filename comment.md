## 一、总则

- 所有代码应具备适当注释，**用来解释“做什么”和“为什么这么做”**，不需要逐行翻译代码。
- 注释应**与代码保持同步**，过时注释要及时修改或删除。
- 禁止出现空注释、无效注释、段落级堆砌性注释。

---

## 二、模块导入注释

**目的**：说明每个导入的模块用途。  

**规范**：
- 标准库、第三方库、自定义模块分类书写；
- 每类模块前统一写注释块。

**示例**：
```python
# 标准库：操作文件系统、时间等
import os
import sys
import datetime

# 第三方库：数据处理和可视化
import numpy as np        # 数值计算
import pandas as pd       # 表格数据处理
import matplotlib.pyplot as plt  # 可视化绘图

# 项目内部模块
from utils.logging import init_logger  # 初始化日志
```

---

## 三、函数/方法注释

**目的**：说明函数的功能、参数、返回值和异常。

**规范**：
- 使用多行文档字符串 `"""..."""`；
- 描述顺序：功能 → 参数 → 返回值 → 异常（可选）→ 示例（可选）；
- 遵循 Google、NumPy 或自定义风格（推荐统一）。

**示例（简化 Google 风格）**：
```python
def compute_average(values: list[float]) -> float:
    """
    计算数值列表的平均值。

    Args:
        values (list[float]): 输入的数值列表。

    Returns:
        float: 平均值。

    Raises:
        ValueError: 如果输入为空列表。
    """
```

---

## 四、复杂逻辑注释

**目的**：解释难以一眼看懂的算法、边界处理、非直观写法等。  

**规范**：
- 注释写在逻辑代码**之前或行尾**；
- 避免解释明显行为，只注释“**为什么这么做**”。

**示例**：
```python
# 使用二分查找优化查找效率，避免 O(n) 遍历
while left < right:
    mid = (left + right) // 2
    if nums[mid] < target:
        left = mid + 1
    else:
        right = mid
```

---

## 五、库函数调用注释

**目的**：说明为什么调用该函数，重点参数的作用，可能的副作用。

**规范**：
- 遇到封装复杂或行为隐含的函数时必须注释；
- 说明关键参数及预期结果。

**示例**：
```python
# 合并用户信息与订单信息，按 user_id 对齐，保留所有用户
merged_df = pd.merge(user_df, order_df, on="user_id", how="left")
```

---

## 六、类注释

**目的**：概述类的设计目的、主要属性、关键方法。

**规范**：
- 使用类 docstring 说明整体作用；
- 可列出主要属性说明。

**示例**：
```python
class UserSession:
    """
    表示用户的会话状态，包括登录时间、操作记录等。

    Attributes:
        user_id (str): 用户唯一 ID
        login_time (datetime): 登录时间
        logs (list[str]): 操作日志列表
    """
```

---

## 七、临时注释（TODO / FIXME）

**目的**：标记未完成任务、已知 bug、技术债务等。

**规范**：
- 使用 `TODO:` 和 `FIXME:` 前缀；
- 可附带责任人和完成时限。

**示例**：
```python
# TODO: @zhangsan 优化此段排序逻辑，当前为 O(n^2)，2025-05 前处理
# FIXME: 当 config 为 None 时此处会抛异常，需加判断
```

---

## 八、注释风格建议

- 避免重复代码逻辑的解释：
  ```python
  i = i + 1  # 把 i 加 1  ❌
  i += 1     # 增加计数 ✅
  ```

- 避免废话型注释：
  ```python
  # 进入主函数
  def main():  # ❌
      ...
  ```

- 避免堆砌注释或“美术字”注释块：

---

