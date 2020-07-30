library(magrittr)


as_matrix <- function(x) {x %>% as.character() %>% as.numeric() %>% as.matrix()}


# credit: Fiona Burlig and Matt Woerman
# this function has been cross checked with Sol's matlab code
# by both Fiona and me
# and it produces consistent outputs
ConleySE <- function(ydata, xdata, latdata, londata, cutoff) {
    # usual setup
    n <- nrow(xdata)
    k <- ncol(xdata)
    betahat <- solve(t(xdata) %*% xdata) %*% t(xdata) %*% ydata
    e <- ydata - xdata %*% betahat
    # loop over all of the spatial units (aka observations)
    meatWeight <- parallel::mclapply(1:n, function(i) {
        # turn longitude & latitude into KMs. 1 deg lat = 111 km, 1 deg lon = 111 km * cos(lat)
        lonscale <- cos(latdata[i] * pi / 180) * 111
        latscale <- 111
        # distance -> use pythagorean theorem! who knew that was useful?
        dist <- as.numeric(sqrt((latscale * (latdata[i] - latdata)) ^ 2
                           + (lonscale * (londata[i] - londata)) ^ 2))
        # set a window var = 1 iff observation j is within cutoff dist of obs i
        window <- as.numeric(dist <= cutoff)
        # this next part is where the magic happens. this thing makes:
        # sum_j(X_iX_j'e_ie_j K(d_{ij})), and we make n of them - one for each i.
        # double transpose here is because R is bad at dealing with 1 x something stuff.
        # we want x_i'; this is an easy way to get it. Now for some dimensions
        # (we want k x k at the end):
        XeeXh <- ((t(t(xdata[i,])) %*% matrix(1, 1, n) * e[i,]) *
                  # k x 1 1 x n 1 x 1
                  (matrix(1, k, 1) %*% (t(e) * t(window)))) %*% xdata
                  # k x 1 1 x n n x k
        return(XeeXh)
    }, mc.cores=parallel::detectCores())
    # phew! Now let's make our sandwich. First, the meat = sum_i what we just made
    meat <- (Reduce("+", meatWeight)) / n
    # and the usual bread
    bread <- solve(t(xdata) %*% xdata)
    # mmmm, delicious sandwich
    sandwich <- n * (t(bread) %*% meat %*% bread)
    # se as per usual
    se <- sqrt(diag(sandwich))
    output <- list(betahat, se)
    names(output) <- c("betahat", "conleySE")
    # print(output)
    return(output)
}


regress_bin <- function(df, col_y, cutoff=3) {
    df <- df %>%
        # cut x into bins
        dplyr::mutate(x_bin=cut(df[['treat_eligible']],
                                breaks=c(-Inf, 0, 1, 2, Inf),
                                labels=c(0, 1, 2, 3)))

    # show distribution
    # print('Distribution of treat_eligible:')
    # print(df %>% dplyr::count(x_bin))

    # regression
    reg <- lfe::felm(
        formula(paste0(col_y, ' ~ factor(x_bin) | factor(eligible) | 0 | lat + lon')),
        data=df %>% tidyr::drop_na(!! col_y),
        keepCX=TRUE)
    # print(summary(reg))

    # calculate conley standard errors
    # with a uniform kernel
    se <- ConleySE(
        ydata=as.matrix(reg$cY),
        xdata=as.matrix(reg$cX),
        latdata=as_matrix(reg$clustervar[['lat']]),
        londata=as_matrix(reg$clustervar[['lon']]),
        cutoff=cutoff  # km
    )

    # collate result for plotting
    res <- tibble::tibble(
        x=c(1, 2, 3), beta=se$betahat, se=se$conleySE)
    res <- rbind(c(0, 0, 0), res)
    return(res)
}


regress_linear <- function(df, col_y, cutoff=3) {

    # regression
    reg <- lfe::felm(
        formula(paste0(col_y, ' ~ treat_eligible | factor(eligible) | 0 | lat + lon')),
        data=df %>% tidyr::drop_na(!! col_y),
        keepCX=TRUE)
    # print(summary(reg))

    # calculate conley standard errors
    # with a uniform kernel
    se <- ConleySE(
        ydata=as.matrix(reg$cY),
        xdata=as.matrix(reg$cX),
        latdata=as_matrix(reg$clustervar[['lat']]),
        londata=as_matrix(reg$clustervar[['lon']]),
        cutoff=cutoff  # km
    )

    # collate result for plotting
    res <- tibble::tibble(x=NA, beta=se$betahat[1, 1], se=se$conleySE)
    return(res)
}


working_dir <- 'output/fig-ate/'

# set color palette
palette <- c('#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6')
# plotting style
g_style <- ggplot2::theme_bw() +
    ggplot2::theme(
        panel.grid.major.x=ggplot2::element_blank(),
        panel.grid.minor=ggplot2::element_blank())

folders <- c('nightlight/', 'building/', 'building/')
col_ys <- c('nightlight', 'area_sum', 'RGB_mean_spline')
titles <- c('Normalized Nightlight Values',
            'Building Footprint (sq meters)',
            'Normalized Roof Reflectance')
y_breaks <- list(c(-0.5, 0, 0.5), c(-50, -25, 0, 25, 50), c(-0.1, 0, 0.1))

for (outcome_i in c(1:length(col_ys))) {
    print(paste0('outcome: ', col_ys[outcome_i]))
    df <- readr::read_csv(
        paste0(working_dir, 'data/', folders[outcome_i], 'main.csv'),
        # to suppress warnings
        col_type=readr::cols())
    main_res <- regress_bin(df=df, col_y=col_ys[outcome_i])
    linear_effect <- regress_linear(df=df, col_y=col_ys[outcome_i])
    # write to file - main results
    main_res_file <- paste0(working_dir, 'data/intermediate/',
                            col_ys[outcome_i], '_main.csv')
    readr::write_csv(rbind(linear_effect, main_res), main_res_file)
    # write to file - placebo results
    placebo_res_file <- paste0(working_dir, 'data/intermediate/',
                               col_ys[outcome_i], '_placebo.csv')
    if (file.exists(placebo_res_file)) {
        placebo_res <- readr::read_csv(
            placebo_res_file,
            # to suppress warnings
            col_type=readr::cols())
    } else {
        placebo_res <- tibble::tibble()
        for (i in c(0:199)) {
            df <- readr::read_csv(
                paste0(working_dir, 'data/', folders[outcome_i],
                       'placebo_', sprintf("%03d", i), '.csv'),
                # to suppress warnings
                col_type=readr::cols())
            res <- regress_bin(df=df, col_y=col_ys[outcome_i])
            placebo_res <- rbind(placebo_res, res %>% dplyr::mutate(iter=i))
        }
        readr::write_csv(placebo_res, placebo_res_file)
    }
    # plotting
    g <- ggplot2::ggplot() +
        ggplot2::geom_line(
            data=placebo_res %>% dplyr::filter(iter < 100),
            ggplot2::aes(x=x, y=beta, group=iter), size=0.5, color='#dddddd', alpha=0.5) +
        ggplot2::geom_line(
            data=main_res, ggplot2::aes(x=x, y=beta), size=1, color='#d7191c') +
        ggplot2::geom_point(
            data=main_res, ggplot2::aes(x=x, y=beta), size=3, color='#d7191c') +
        ggplot2::geom_errorbar(
            data=main_res, ggplot2::aes(x=x, y=beta, ymin=beta - 1.96 * se, ymax=beta + 1.96 * se),
            color='#d7191c', width=.3) +
        ggplot2::scale_x_continuous(
            name='Cash infusion per 0.012 sq km',
            breaks=c(0, 1, 2, 3),
            labels=c('$0', '$1000', '$2000', '>$2000')) +
        ggplot2::scale_y_continuous(
            name='',
            breaks=y_breaks[[outcome_i]]) +
        ggplot2::ggtitle(paste0("Treatment Effect on ", titles[outcome_i], ":\n",
                                round(linear_effect$beta[1], 3), ", 95% CI: [",
                                round(linear_effect$beta[1] - 1.96 * linear_effect$se[1], 3), ", ",
                                round(linear_effect$beta[1] + 1.96 * linear_effect$se[1], 3), "]")) +
        g_style
    ggplot2::ggsave(paste0(working_dir, col_ys[outcome_i], '_ate.pdf'), g, width=5, height=3)
}
