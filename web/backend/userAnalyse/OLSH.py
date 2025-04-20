import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # 用于后处理噪声点
from sklearn.cluster import OPTICS


class OLsh:
    def __init__(self, num_tables=10, num_hashes=12, eps=0.5, min_samples=100):
        self.params = {  # LSH 参数和 OPTICS min_samples
            "num_tables": num_tables,
            "num_hashes": num_hashes,
            "eps": eps,
            "min_samples": min_samples,
        }
        # LSH 结构: List[Cluster][Table] -> Dict[HashKey -> List[(user_id, vector)]]
        self.hash_tables: list[list[dict[tuple, list[tuple[any, np.ndarray]]]]] = []
        self.cluster_centers: list[np.ndarray] = []
        self.hash_functions: list[list[list[tuple[np.ndarray, float]]]] = []
        # Fallback 数据: List[Cluster] -> List[(user_id, vector)]
        self.raw_cluster_data_with_ids: list[list[tuple[any, np.ndarray]]] = []
        self._index_built = False
        self.labels: np.ndarray | None = None  # 最终（无噪声）标签
        self.user_ids: list | None = (
            None  # 对应 labels 的用户 ID 列表 (主要用于 save/load 验证)
        )

    def _hash_function(self, vec, a, b, w=3.0):
        # 简单的 LSH 哈希函数
        return int(np.floor((np.dot(vec, a) + b) / w))

    # 修改 build_index 接收 user_ids, 并在索引中存储 (id, vec)
    def build_index(self, data: np.ndarray, user_ids: list):
        """构建 OPTICS 聚类（带噪声点后处理）和 LSH 索引 (存储用户ID和向量)。"""
        if len(data) != len(user_ids):
            raise ValueError("数据点数量与用户ID数量不匹配！")

        print(f"开始构建索引，输入数据维度: {data.shape}...")
        current_min_samples = self.params["min_samples"]
        current_xi = 0.05  # 可以作为参数传入或在此调整
        print(f"使用 OPTICS 参数: min_samples={current_min_samples}, xi={current_xi}")

        # --- Step 1: 初始 OPTICS ---
        optics = OPTICS(
            min_samples=current_min_samples, xi=current_xi, cluster_method="xi"
        )
        initial_labels = optics.fit_predict(data)
        initial_valid_labels = sorted(list(set(initial_labels) - {-1}))
        num_initial_clusters = len(initial_valid_labels)
        num_noise_points = np.sum(initial_labels == -1)
        print(
            f"OPTICS 初始聚类：找到 {num_initial_clusters} 个有效聚类，{num_noise_points} 个噪声点。"
        )

        # --- Step 2: 后处理 - 分配噪声点 ---
        final_labels = np.copy(initial_labels)
        if num_initial_clusters > 0 and num_noise_points > 0:
            # ... (噪声分配逻辑不变) ...
            print(f"开始后处理：分配 {num_noise_points} 个噪声点...")
            noise_indices = np.where(initial_labels == -1)[0]
            initial_centers_list = [
                np.mean(data[initial_labels == label], axis=0)
                for label in initial_valid_labels
                if np.any(initial_labels == label)
            ]
            if not initial_centers_list:
                print("[警告] 无有效中心，无法分配噪声")
            else:
                centers_array = np.array(initial_centers_list)
                valid_labels_map = {
                    label: i
                    for i, label in enumerate(initial_valid_labels)
                    if np.any(initial_labels == label)
                }  # map label to index in centers_array
                valid_labels_list = list(valid_labels_map.keys())

                dists = cdist(data[noise_indices], centers_array)
                nearest_center_indices = np.argmin(dists, axis=1)
                assigned_count = 0
                for i, noise_idx in enumerate(noise_indices):
                    center_list_idx = nearest_center_indices[i]
                    # Make sure index is valid before assignment
                    if center_list_idx < len(valid_labels_list):
                        assigned_label = valid_labels_list[center_list_idx]
                        final_labels[noise_idx] = assigned_label
                        assigned_count += 1
                print(f"成功分配 {assigned_count} 个噪声点。")
                num_noise_after = np.sum(final_labels == -1)
                print(f"后处理后噪声点: {num_noise_after}")
        else:
            print("无需噪声点后处理。")
        self.labels = final_labels

        # --- Step 3: 基于最终标签构建 LSH (存储 ID, Vec) ---
        print("基于最终标签构建LSH索引 (存储 ID, Vec)...")
        self.cluster_centers = []
        self.raw_cluster_data_with_ids = []
        self.hash_tables = []
        self.hash_functions = []
        self._index_built = False

        final_valid_labels = sorted(list(set(self.labels) - {-1}))
        num_final_clusters = len(final_valid_labels)
        if num_final_clusters == 0:
            print("[警告] 最终无有效聚类")
            self._index_built = True
            return

        data_dim = data.shape[1]
        # 生成与最终簇数匹配的哈希参数
        all_hash_params = [
            [
                [
                    (np.random.randn(data_dim), np.random.rand())
                    for _ in range(self.params["num_hashes"])
                ]
                for _ in range(self.params["num_tables"])
            ]
            for _ in range(num_final_clusters)
        ]

        lsh_cluster_count = 0
        for i, label in enumerate(final_valid_labels):
            mask = self.labels == label
            cluster_indices = np.where(mask)[0]  # 获取属于该簇的原始数据索引

            if cluster_indices.size > 0:
                cluster_data_subset = data[cluster_indices]
                self.cluster_centers.append(np.mean(cluster_data_subset, axis=0))

                # 准备 (id, vec) 元组列表用于 LSH 和 fallback
                id_vec_list = [(user_ids[idx], data[idx]) for idx in cluster_indices]
                self.raw_cluster_data_with_ids.append(id_vec_list)

                # 构建 LSH 表
                cluster_tables_for_c = []
                # 使用 lsh_cluster_count 作为 all_hash_params 的索引，因为它对应实际构建的簇
                cluster_hash_params_for_c = all_hash_params[lsh_cluster_count]

                for t, hash_params in enumerate(cluster_hash_params_for_c):
                    # 使用 defaultdict 提高一点效率
                    table = defaultdict(list)
                    for user_id, point_vec in id_vec_list:  # 直接迭代 (id, vec) 对
                        hash_key = tuple(
                            self._hash_function(point_vec, a, b) for a, b in hash_params
                        )
                        table[hash_key].append((user_id, point_vec))  # 存入 (id, vec)
                    cluster_tables_for_c.append(dict(table))  # 转回普通 dict (如果需要)

                self.hash_tables.append(cluster_tables_for_c)
                self.hash_functions.append(cluster_hash_params_for_c)
                lsh_cluster_count += 1  # 增加成功构建LSH的簇计数
            else:
                print(f"[警告] 最终聚类标签 {label} 无数据点，跳过LSH构建。")

        self._index_built = True
        self.user_ids = user_ids  # 保存引用
        print(f"LSH 索引构建完成，覆盖 {lsh_cluster_count} 个最终聚类。")

    # _save_cluster_assignments_json 不变
    def _save_cluster_assignments_json(self, json_filename: Path, user_ids: list):
        # ... (代码不变) ...
        if self.labels is None:
            print("[警告] 标签信息不存在")
            return
        if len(self.labels) != len(user_ids):
            print("[错误] 标签/ID数量不匹配")
            return
        print(f"正在将最终聚类分配结果保存到 {json_filename}...")
        cluster_assignments = defaultdict(list)
        num_noise = 0
        for idx, label in enumerate(self.labels):
            key = f"cluster_{label}" if label != -1 else "noise_unexpected"
            num_noise += label == -1
            cluster_assignments[key].append(user_ids[idx])
        if num_noise > 0:
            print(f"[警告] 保存JSON时发现 {num_noise} 个噪声点!")
        try:
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(cluster_assignments, f, indent=4, ensure_ascii=False)
            print(f"最终聚类分配结果成功保存到 {json_filename}。")
        except Exception as e:
            print(f"[错误] 保存JSON失败: {e}")

    # save 方法保存更新后的结构名
    def save(self, filename, user_ids: list):
        if not self._index_built:
            print("[警告] 索引未构建")
            return
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        index_data = {
            "cluster_centers": self.cluster_centers,
            "raw_cluster_data_with_ids": self.raw_cluster_data_with_ids,  # 保存新结构
            "hash_tables": self.hash_tables,  # 桶内是(id,vec)
            "hash_functions": self.hash_functions,
            "params": self.params,
            "labels": self.labels,
            "user_ids": user_ids,
        }
        try:
            joblib.dump(index_data, filepath, compress=3)
            print(f"最终索引成功保存到 {filepath}。")
            json_filepath = filepath.with_suffix(".json")
            self._save_cluster_assignments_json(json_filepath, user_ids)
        except Exception as e:
            print(f"[错误] 保存索引文件 {filepath} 失败: {e}")

    # load 方法加载更新后的结构名
    def load(self, filename):
        filepath = Path(filename)
        if not filepath.exists():
            return False
        try:
            index_data = joblib.load(filepath)
            self.cluster_centers = index_data["cluster_centers"]
            self.raw_cluster_data_with_ids = index_data.get(
                "raw_cluster_data_with_ids", []
            )  # 加载新结构
            self.hash_tables = index_data["hash_tables"]  # 桶内是(id,vec)
            self.hash_functions = index_data["hash_functions"]
            self.params = index_data.get("params", self.params)
            self.labels = index_data.get("labels", None)
            self.user_ids = index_data.get("user_ids", None)
            self._index_built = True
            print(f"索引成功从 {filepath} 加载 ({len(self.cluster_centers)} 个聚类)。")
            # if self.labels is not None and self.user_ids is not None and len(self.labels) != len(self.user_ids):
            #      print(f"[警告] 加载的 labels/user_ids 长度不匹配！")
            return True
        except Exception as e:
            print(f"[错误] 从 {filepath} 加载索引失败: {e}")
            self.__init__(**self.params)
            return False

    # 修改 query 方法返回 (user_id, score) 列表
    def query(
        self, target: np.ndarray, k: int = 5, theta: float = 1.0
    ) -> list[tuple[any, float]]:
        """使用 LSH 查询索引以查找最近邻居的用户ID和相似度得分。"""
        if not self._index_built:
            raise RuntimeError("索引未构建或加载。")
        if not self.cluster_centers:
            return []

        # 1. 查找候选聚类
        centers_array = np.array(self.cluster_centers)
        # 计算 target 与所有 cluster_centers 的距离
        distances_to_centers = np.linalg.norm(centers_array - target, axis=1)
        # 找到距离小于等于 theta 的聚类索引 (这些索引对应 self.cluster_centers 等列表)
        candidate_cluster_indices = np.where(distances_to_centers <= theta)[0]
        if candidate_cluster_indices.size == 0:
            return []  # 没有候选聚类，直接返回

        neighbors = []
        retrieved_user_ids = set()  # 用于跟踪已添加的用户ID，确保返回结果唯一

        # 2. 在候选聚类中搜索
        for (
            cluster_idx
        ) in candidate_cluster_indices:  # cluster_idx 是 self.hash_tables 等列表的索引
            # 安全检查：确保索引有效
            if cluster_idx >= len(self.hash_tables):
                continue

            found_in_bucket = False  # 标记是否在 LSH 桶中找到过此簇的邻居

            # 2a. 搜索 LSH 哈希桶
            for table, hash_params in zip(
                self.hash_tables[cluster_idx], self.hash_functions[cluster_idx]
            ):
                target_hash_key = tuple(
                    self._hash_function(target, a, b) for a, b in hash_params
                )
                # 获取桶中的候选者列表，每个元素是 (user_id, vector)
                candidates_in_bucket = table.get(target_hash_key, [])
                if candidates_in_bucket:
                    found_in_bucket = True  # 标记在此簇的桶中找到了
                    for user_id, candidate_vec in candidates_in_bucket:
                        # 如果这个用户ID还没被加到结果里
                        if user_id not in retrieved_user_ids:
                            distance = np.linalg.norm(target - candidate_vec)
                            # 计算相似度得分（距离越近得分越高）
                            score = 1.0 / (1.0 + distance)
                            neighbors.append((user_id, score))  # 添加 (user_id, score)
                            retrieved_user_ids.add(user_id)  # 标记此用户ID已添加

            # 2b. 如果 LSH 未命中，进行 Fallback 扫描
            # 使用 self.raw_cluster_data_with_ids 进行 fallback
            if not found_in_bucket and cluster_idx < len(
                self.raw_cluster_data_with_ids
            ):
                # print(f"[DEBUG] Fallback 扫描聚类索引 {cluster_idx}") # 可选的调试信息
                # 遍历该簇存储的 (user_id, vector) 对
                for user_id, candidate_vec in self.raw_cluster_data_with_ids[
                    cluster_idx
                ]:
                    if user_id not in retrieved_user_ids:
                        distance = np.linalg.norm(target - candidate_vec)
                        score = 1.0 / (1.0 + distance)
                        neighbors.append((user_id, score))
                        retrieved_user_ids.add(user_id)

        # 3. 对所有找到的邻居按得分排序，返回前 k 个
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:k]


# --- 独立函数：查找最近的聚类中心 ---
# ... (find_most_similar_cluster 函数不变) ...
def find_most_similar_cluster(
    target_vector: np.ndarray, olsh_instance: OLsh
) -> int | None:
    if not olsh_instance._index_built or not olsh_instance.cluster_centers:
        return None
    centers_array = np.array(olsh_instance.cluster_centers)
    distances = np.linalg.norm(centers_array - target_vector, axis=1)
    # return int(np.argmin(distances))
    # 获取距离最小的 5 个索引（升序排列）
    top5_indices = np.argsort(distances)[:5]
    return [int(idx) for idx in top5_indices]


def find_most_similar_user(target_user_vector: np.ndarray, olsh_instance: OLsh):
    if not olsh_instance._index_built or not olsh_instance.cluster_centers:
        return None

    results = olsh_instance.query(
        target_user_vector, k=5, theta=3.0
    )  # theta 可能需调整
    return results


# --- 主执行部分 ---
if __name__ == "__main__":
    min_samples_param = 15
    index_file = Path("userAnalyse/olsh_index.joblib")
    csv_file = Path("userAnalyse/output8.csv")

    # --- Load Data ---
    try:
        print(f"加载数据: {csv_file}...")
        df = pd.read_csv(csv_file)
        feature_columns = [f"feature_{i}" for i in range(8)]
        user_id_column = "hash_sec_uid"
        if not all(col in df.columns for col in feature_columns):
            raise ValueError("CSV缺少特征列")
        if user_id_column not in df.columns:
            raise ValueError(f"CSV缺少ID列 '{user_id_column}'")
        data = df[feature_columns].values
        user_ids = df[user_id_column].tolist()
        if len(data) == 0:
            raise ValueError("CSV数据为空")
        if len(data) != len(user_ids):
            raise ValueError("数据/ID数量不匹配")
        print(f"成功加载 {len(data)} 条数据。")
    except Exception as e:
        print(f"[错误] 加载CSV出错: {e}")
        exit()

    # --- Init OLSH ---
    olsh = OLsh(min_samples=min_samples_param)

    # --- Load or Build Index ---
    if not olsh.load(index_file):
        print("索引加载失败，重新构建 (含后处理)...")
        # 传入 user_ids
        olsh.build_index(data, user_ids)
        if olsh._index_built and olsh.cluster_centers:
            # 传入 user_ids
            olsh.save(index_file, user_ids)
        elif not olsh.cluster_centers:
            print("[信息] 未找到聚类，不保存。")

    # --- Query Example ---
    if olsh._index_built and olsh.cluster_centers:
        print("-" * 30)
        print(f"索引就绪 ({len(olsh.cluster_centers)} 个最终聚类)。")
        json_output_file = index_file.with_suffix(".json")
        print(f"最终分配保存在: {json_output_file}")
        print("-" * 30)

        if len(data) > 0:
            target_user_row_index = np.random.randint(len(data))
            target_user_vector = data[target_user_row_index]
            target_user_id = user_ids[target_user_row_index]
            print(
                f"\n查询目标: 用户 ID '{target_user_id}' (行 {target_user_row_index})"
            )

            # 查找最近簇 (不变)
            # ... (find_most_similar_cluster call & print) ...
            similar_cluster_idx = find_most_similar_cluster(target_user_vector, olsh)
            if similar_cluster_idx is not None:
                print(f"  -> 距离聚类 {similar_cluster_idx} 中心最近")
            if olsh.labels is not None:
                print(f"  -> 最终分配标签: {olsh.labels[target_user_row_index]}")

            # 执行 LSH 查询, 结果是 (user_id, score)
            print("LSH 查询近邻 (k=5, theta=3.0)...")
            results = olsh.query(target_user_vector, k=5, theta=3.0)  # theta 可能需调整

            print(f"找到 {len(results)} 个 LSH 近邻:")
            if results:
                # 直接打印 user_id 和 score
                for i, (neighbor_user_id, score) in enumerate(results):
                    print(
                        f"  {i + 1}: 用户ID = {neighbor_user_id}, 相似度 = {score:.4f}"
                    )
            else:
                print("  未找到 LSH 近邻。")
        else:
            print("\n无数据无法查询。")
    else:
        print("\n索引未就绪。")
