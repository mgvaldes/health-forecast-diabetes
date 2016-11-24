library(data.table)

for (i in 1:22) {
  dir.create(paste0("chr", i))

  system(paste0("/opt/plink-beta-3.38/plink --bfile GCAT_genome_data --chr ", i, " --make-bed --out chr", i, "/GCAT_genome_data_chr", i))
  system(paste0("/opt/plink-beta-3.38/plink --bfile chr", i, "/GCAT_genome_data_chr", i, " --recode AD --out chr", i, "/GCAT_genome_data_chr", i))

  diabetes.genomic.data <- fread(paste0("chr", i, "/GCAT_genome_data_chr", i, ".raw"), header = TRUE, data.table = FALSE)

  diabetes.identifier.data <- fread("genotyped.csv", header = TRUE, data.table = FALSE)

  diabetes.epidemiological.data <- fread("binary.csv", header = TRUE, data.table = FALSE)

  ###### Eliminate duplicates in diabetes.identifier.data
  diabetes.identifier.data.entity.id.duplicates.indexes <- which(duplicated(diabetes.identifier.data[, "entity_id"]) == TRUE)
  diabetes.identifier.data <- diabetes.identifier.data[-diabetes.identifier.data.entity.id.duplicates.indexes, ]

  ##### entity_id identifiers of observations that match diabetes.genomic.data's IDD with diabetes.identifier.data's Sample.name.
  diabetes.identifier.data.entity.id <- diabetes.identifier.data[which((diabetes.identifier.data[, "Sample.name"] %in% diabetes.genomic.data[, "IID"]) == TRUE), "entity_id"]

  ###### Eliminate element in diabetes.genomic.data that it is not in diabetes.identifier.data
  diabetes.genomic.data <- diabetes.genomic.data[-which(!(diabetes.genomic.data[, "IID"] %in% diabetes.identifier.data[, "Sample.name"]) == TRUE), ]

  ###### Eliminate duplicates in diabetes.epidemiological.data
  diabetes.epidemiological.data.entity.id.duplicates.indexes <- which(duplicated(diabetes.epidemiological.data[, "entity_id"]) == TRUE)
  diabetes.epidemiological.data <- diabetes.epidemiological.data[-diabetes.epidemiological.data.entity.id.duplicates.indexes, ]

  filtered.diabetes.epidemiological.data <- diabetes.epidemiological.data[, !(colnames(diabetes.epidemiological.data) %in% c('FECHA_NACIMIENTO', 'MUNICIPIO_RESIDENCIA', 'MUNICIPIO_NACIMIENTO', 'PROVINCIA_RESIDENCIA', 'PAIS_NACIMIENTO'))]

  filtered.diabetes.epidemiological.data <- filtered.diabetes.epidemiological.data[match(diabetes.identifier.data[match(diabetes.genomic.data[, "IID"], diabetes.identifier.data[, "Sample.name"]), "entity_id"], filtered.diabetes.epidemiological.data[, "entity_id"]), ]

  filtered.diabetes.genomic.data <- diabetes.genomic.data[, !(colnames(diabetes.genomic.data) %in% c("FID", "PAT", "MAT", "SEX", "PHENOTYPE"))]

  merged.genomic.epidemiological.data <- cbind(filtered.diabetes.epidemiological.data, filtered.diabetes.genomic.data)
  merged.genomic.epidemiological.data <- merged.genomic.epidemiological.data[, !(colnames(merged.genomic.epidemiological.data) %in% c("entity_id", "sampleType", "IID"))]

  write.table(merged.genomic.epidemiological.data, file = paste0("chr", i, "/genomic_epidemiological.csv"), row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

  rm(list = ls())
}

# Replace NA's with -1 values.
for (i in 2:22) {
  system(paste0("sed -i -- 's/NA/-1/g' chr", i, "/genomic_epidemiological.csv"))
}