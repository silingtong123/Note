# Parquet数据格式

### 简介
- 列式存储一个类型包含嵌套结构的数据集
- 行存储和列储存， 一个连续的内存块为例：
  - 按行储存： Lilei 18 173*  Hanmm ...
  - 按列储存： Lilei  Hanmm Tom 18 17 ...

| name | age | num|
| --- | --- | ---|
| Lilei | 18 | 173* |
| Hanmm | 17 | 135 *|
| Tom | 28 |  151 * |
- 当列表元素为嵌套值时：在Parquet里面，保存嵌套结构的方式是把所有字段打平以后顺序存储，由于嵌套结构里可能有动态数组，即length不固定，无法区别record边界， 所以引入repetition level和definition level。这两个值会保存额外的信息，可以用来重构出数据原本的结构
- Parquet文件对每个value，都同时保存了它们的repetition level和definition level
- Parquet文件结构图：
- Header --> Data --> index --> Footer
  - Data: Row Group(list)
  - Row Group: Column chunk(list) --->将原始数据水平切分为Row Group，其包括一些行的所有列
  - Column chunk: Page(list) ---> 将Row Group垂直切分，每个Column储存该group的某一列 
  - Page: 将Column水分切分为1M大小的page，其结构为：page header --> repetition levels --> definition levels --> values
  - Header： magic num
  - Index: Max-Min索引记录每个page的，方便索引减少IO;两种索引：Max-Min索引，BloomFilter
  - Footer： 元数据的大本营，，Block的offset和size，Column Chunk的offset和size等所有重要的元数据。另外Footer还承担了整个文件入口的职责，通常Footer信息被第一步读取


### Deeprec parquet
- GraphDefBuilderWrapper: 用于序列化数据集图。 
  - AddScalar： 给graph 添加一个scalar const节点
  - AddDataset：将对应于 `DatasetType` 的节点添加到 Graph。