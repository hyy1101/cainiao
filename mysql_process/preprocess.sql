/*建表*/
CREATE DATABASE cainiao;
CREATE TABLE `cainiao`.`item_feature`( `date` DATE, `item_id` BIGINT, `cate_id` BIGINT, `cate_level_id` BIGINT, `brand_id` BIGINT, `supplier_id` BIGINT, `pv_ipv` BIGINT, `pv_uv` BIGINT, `cart_ipv` BIGINT, `cart_uv` BIGINT, `collect_uv` BIGINT, `num_gmv` BIGINT, `amt_gmv` DOUBLE, `qty_gmv` BIGINT, `unum_gmv` BIGINT, `amt_alipay` DOUBLE, `num_alipay` BIGINT, `qty_alipay` BIGINT, `unum_alipay` BIGINT, `ztc_pv_ipv` BIGINT, `tbk_pv_ipv` BIGINT, `ss_pv_ipv` BIGINT, `jhs_pv_ipv` BIGINT, `ztc_pv_uv` BIGINT, `tbk_pv_uv` BIGINT, `ss_pv_uv` BIGINT, `jhs_pv_uv` BIGINT, `num_alipay_njhs` BIGINT, `amt_alipay_njhs` DOUBLE, `qty_alipay_njhs` BIGINT, `unum_alipay_njhs` BIGINT ) ENGINE=MYISAM;
CREATE TABLE `cainiao`.`item_store_feature`( `date` DATE, `item_id` BIGINT, `store_code` BIGINT, `cate_id` BIGINT, `cate_level_id` BIGINT, `brand_id` BIGINT, `supplier_id` BIGINT, `pv_ipv` BIGINT, `pv_uv` BIGINT, `cart_ipv` BIGINT, `cart_uv` BIGINT, `collect_uv` BIGINT, `num_gmv` BIGINT, `amt_gmv` DOUBLE, `qty_gmv` BIGINT, `unum_gmv` BIGINT, `amt_alipay` DOUBLE, `num_alipay` BIGINT, `qty_alipay` BIGINT, `unum_alipay` BIGINT, `ztc_pv_ipv` BIGINT, `tbk_pv_ipv` BIGINT, `ss_pv_ipv` BIGINT, `jhs_pv_ipv` BIGINT, `ztc_pv_uv` BIGINT, `tbk_pv_uv` BIGINT, `ss_pv_uv` BIGINT, `jhs_pv_uv` BIGINT, `num_alipay_njhs` BIGINT, `amt_alipay_njhs` DOUBLE, `qty_alipay_njhs` BIGINT, `unum_alipay_njhs` BIGINT ) ENGINE=MYISAM;
USE cainiao;
/*导入数据1*/
show tables;
DROP TABLE if EXISTS item_store_feature;
LOAD DATA LOCAL INFILE'/Users/hyy/Downloads/CAINIAO/item_store_feature2.csv' IGNORE INTO TABLE item_store_feature FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '' ESCAPED BY '' LINES TERMINATED BY '\n';

/*导入数据2*/
LOAD DATA LOCAL INFILE '/Users/hyy/Downloads/CAINIAO/item_feature2.csv'
IGNORE INTO TABLE item_feature
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '' ESCAPED BY ''
LINES TERMINATED BY '\n';

/*查看数据2*/
SELECT * FROM item_feature LIMIT 10;

/*删除双11，12*/
DELETE FROM item_feature WHERE DATE='2015/11/11' OR DATE='2015/12/12';
DELETE FROM item_store_feature WHERE DATE='2015/11/11' OR DATE='2015/12/12';

/*检验下是否删除双11，12*/
SELECT * FROM item_feature WHERE DATE='2015-11-11'
LIMIT 10;