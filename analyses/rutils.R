library(matrixStats)

# Many thanks to https://www.r-bloggers.com/2016/02/a-faster-scale-function/ for this function

colScale = function(x,
                    center = TRUE,
                    scale = TRUE,
                    add_attr = TRUE,
                    rows = NULL,
                    cols = NULL) {
    if (!is.null(rows) && !is.null(cols)) {
        x <- x[rows, cols, drop = FALSE]
    } else if (!is.null(rows)) {
        x <- x[rows, , drop = FALSE]
    } else if (!is.null(cols)) {
        x <- x[, cols, drop = FALSE]
    }
    ################
    # Get the column means
    ################
    cm = colMeans(x, na.rm = TRUE)
    ################
    # Get the column sd
    ################
    if (scale) {
        csd = colSds(x, center = cm, na.rm=TRUE)
    } else {
        # just divide by 1 if not
        csd = rep(1, length = length(cm))
    }
    if (!center) {
        # just subtract 0
        cm = rep(0, length = length(cm))
    }
    x = t( (t(x) - cm) / csd )
    if (add_attr) {
        if (center) {
            attr(x, "scaled:center") <- cm
        }
        if (scale) {
            attr(x, "scaled:scale") <- csd
        }
    }
    return(x)
}