package smile

import org.apache.commons.csv.CSVFormat
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import smile.regression.GradientTreeBoost

fun main() {
    // Загрузка датасета экономических индикаторов
    val csvFormat = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .setDelimiter(',')
        .build()

    val ds = Read.csv(
        "src/main/resources/economic_indicators_dataset_2010_2023.csv",
        csvFormat
    )
// Удаляем нечисловые столбцы (Date, Country)
    val dataset = ds.drop("Date", "Country")
//    println(dataset)

    // Формула для целевой переменной (прогнозируем фондовый индекс)
    val formula = Formula.lhs("Stock Index Value")

    // Кросс-валидация для регрессии
    val res = CrossValidation.regression(
        10, formula, dataset,
        { formula, data -> GradientTreeBoost.fit(formula, data) }
    )

    // Выводим результат кросс-валидации
    println("\nРезультаты кросс-валидации (Random Forest - регрессия):")
    println(res)
}