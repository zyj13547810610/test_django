import subprocess
import re
import time
from collections import namedtuple
from typing import Dict, Any, Optional

# 数据结构定义
MemoryInfo = namedtuple("MemoryInfo", "total used free")


class SimpleNPUMonitor:
    def __init__(self, cache_ttl: float = 0.8):
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._cache_ttl = cache_ttl
        self._cache_ts = 0.0

        # 修正后的正则表达式
        self._re_l1 = re.compile(
            r"^\|\s*(\d+)\s+(\S+).*?\|\s*(\S+)\s+\|\s*([\d.]+)\s+(\d+)"
        )
        self._re_l2 = re.compile(r"^\|\s*\d+\s+\|\s+([0-9A-Fa-f:.]+).*?\|\s+(\d+)")

    def _update_cache(self) -> None:
        """更新缓存数据"""
        if time.time() - self._cache_ts < self._cache_ttl:
            return

        try:
            # 执行npu-smi命令
            raw = subprocess.run(
                ["npu-smi", "info"], text=True, capture_output=True, timeout=3
            ).stdout.splitlines()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return

        data: Dict[int, Dict[str, Any]] = {}
        cur_id = -1

        for line in raw:
            # 匹配设备基本信息行（第一行：NPU ID + 设备名称）
            m1 = self._re_l1.match(line)
            if m1:
                npu_id, name, health, power, temp = m1.groups()
                cur_id = int(npu_id)
                d = data.setdefault(cur_id, {})
                d.update(
                    {
                        "name": name,
                        "health": health,
                        "power": float(power) * 1000,  # 转换为毫瓦
                        "temp": int(temp),
                    }
                )
                continue

            # 匹配内存和AI Core利用率信息（第二行）
            m2 = self._re_l2.match(line)
            if m2 and cur_id >= 0:
                bus_id, aicore_util = m2.groups()

                # 提取HBM内存使用情况 (格式: used/total)
                memory_matches = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                if memory_matches:
                    # 取最后一个匹配（HBM-Usage部分）
                    h_used, h_total = map(int, memory_matches[-1])

                    d = data.setdefault(cur_id, {})
                    d.update(
                        {
                            "bus_id": bus_id,
                            "hbm_used": h_used * 1024 * 1024,  # 转换为字节
                            "hbm_total": h_total * 1024 * 1024,  # 转换为字节
                            "aicore_util": int(aicore_util),
                        }
                    )

        self._cache.clear()
        self._cache.update(data)
        self._cache_ts = time.time()

    def get_memory_info(self, device_id: int) -> Optional[MemoryInfo]:
        """获取指定设备的内存信息"""
        self._update_cache()

        device_data = self._cache.get(device_id, {})
        total = device_data.get("hbm_total", 0)
        used = device_data.get("hbm_used", 0)
        free = total - used if total > 0 else 0

        return MemoryInfo(total=total, used=used, free=free)

    def get_aicore_utilization(self, device_id: int) -> Optional[int]:
        """获取指定设备的AI Core利用率"""
        self._update_cache()

        device_data = self._cache.get(device_id, {})
        return device_data.get("aicore_util")

    def get_device_count(self) -> int:
        """获取设备数量"""
        self._update_cache()
        return len(self._cache)

    def get_all_devices_info(self) -> Dict[int, Dict[str, Any]]:
        """获取所有设备的内存和AI Core信息"""
        self._update_cache()

        result = {}
        for device_id in self._cache.keys():
            memory_info = self.get_memory_info(device_id)
            aicore_util = self.get_aicore_utilization(device_id)

            result[device_id] = {
                "memory": {
                    "total_mb": (
                        memory_info.total // (1024 * 1024) if memory_info else 0
                    ),
                    "used_mb": memory_info.used // (1024 * 1024) if memory_info else 0,
                    "free_mb": memory_info.free // (1024 * 1024) if memory_info else 0,
                    "usage_percent": (
                        round(100 * memory_info.used / memory_info.total, 1)
                        if memory_info and memory_info.total > 0
                        else 0
                    ),
                },
                "aicore_utilization_percent": aicore_util or 0,
            }

        return result


# 使用示例
if __name__ == "__main__":
    monitor = SimpleNPUMonitor()

    # 获取所有设备信息
    devices_info = monitor.get_all_devices_info()

    for device_id, info in devices_info.items():
        print(f"NPU {device_id}:")
        print(
            f"  显存: {info['memory']['used_mb']}/{info['memory']['total_mb']} MB "
            f"({info['memory']['usage_percent']}%)"
        )
        print(f"  AI Core利用率: {info['aicore_utilization_percent']}%")
        print()
