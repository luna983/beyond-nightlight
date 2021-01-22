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


regress_bin <- function(df, col_y, col_x='treat_eligible', cutoff=3, compute_se=T) {
    df <- df %>%
        # cut x into bins
        dplyr::mutate(x_bin=cut(df[[col_x]],
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

    # calculate conley standard errors
    # with a uniform kernel
    if (compute_se) {
        se <- ConleySE(
            ydata=as.matrix(reg$cY),
            xdata=as.matrix(reg$cX),
            latdata=as_matrix(reg$clustervar[['lat']]),
            londata=as_matrix(reg$clustervar[['lon']]),
            cutoff=cutoff  # km
        )
        conley_se <- se$conleySE
    } else {
        conley_se <- NA
    }

    # collate result for plotting
    res <- tibble::tibble(
        x=c(1, 2, 3), beta=reg$coefficients, se=conley_se)
    res <- rbind(c(0, 0, 0), res)
    return(res)
}


regress_linear <- function(df, col_y, col_x='treat_eligible', cutoff=3) {

    # regression
    reg <- lfe::felm(
        formula(paste0(col_y, ' ~ ', col_x, ' | factor(eligible) | 0 | lat + lon')),
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


input_dir <- 'data/Siaya/Merged/main_res0.0010.csv'
output_dir <- 'output/fig-ate/'

# set color palette
palette <- c('#820000', '#ea0000', '#fff4da', '#5d92c4', '#070490')
# plotting style
g_style <- ggplot2::theme_bw() +
    ggplot2::theme(
        panel.border=ggplot2::element_blank(),
        panel.grid.major=ggplot2::element_blank(),
        panel.grid.minor=ggplot2::element_blank(),
        axis.title=ggplot2::element_text(size=12),
        axis.text=ggplot2::element_text(size=12),
        plot.title=ggplot2::element_text(size=12),
        axis.ticks.length=ggplot2::unit(.07, "in"),
        axis.ticks.x=ggplot2::element_blank())

col_ys <- c('nightlight', 'area_sum', 'tin_area_sum')
titles <- c('Night Light',
            'Building Footprint (sq meters)',
            'Tin-roof Area (sq meters)')
y_breaks <- list(
    c(-0.05, 0, 0.05), c(-25, 0, 25, 50), c(-25, 0, 25, 50, 75))
y_lims <- list(
    c(-0.055, 0.055), c(-30, 55), c(-30, 80))
x_lim <- c(-0.2, 3.5)
placebo <- T

# load data
df <- readr::read_csv(
    input_dir,
    # to suppress warnings
    col_type=readr::cols())
n_iter <- stringr::str_detect(colnames(df),
                              'treat_eligible_placebo') %>% sum()
for (outcome_i in c(1:length(col_ys))) {
    print('----------------------------------')
    print(paste0('outcome: ', col_ys[outcome_i]))
    print('Main effect estimation...')
    main_res_file <- paste0(output_dir, 'cache/',
                            col_ys[outcome_i], '_main.csv')
    if (file.exists(main_res_file)) {
        main_res_merged <- readr::read_csv(
            main_res_file,
            # to suppress warnings
            col_type=readr::cols())
        main_res <- main_res_merged %>% dplyr::filter(!is.na(x))
        linear_effect <- main_res_merged %>% dplyr::filter(is.na(x))
    } else {
        main_res <- regress_bin(df=df, col_y=col_ys[outcome_i])
        linear_effect <- regress_linear(df=df, col_y=col_ys[outcome_i])
        # write to file - main results
        readr::write_csv(rbind(linear_effect, main_res), main_res_file)
    }
    print('Placebo effect estimation...')
    if (placebo) {
        # write to file - placebo results
        placebo_res_file <- paste0(output_dir, 'cache/',
                                   col_ys[outcome_i], '_placebo.csv')
        if (file.exists(placebo_res_file)) {
            placebo_res <- readr::read_csv(
                placebo_res_file,
                # to suppress warnings
                col_type=readr::cols())
        } else {
            placebo_res <- tibble::tibble()
            for (i in c(0:(n_iter - 1))) {
                if (i %% 10 == 0) {print('++++++++++')}
                res <- regress_bin(df=df, col_x=sprintf('treat_eligible_placebo%02d', i),
                                   col_y=col_ys[outcome_i], compute_se=F)
                placebo_res <- rbind(placebo_res, res %>% dplyr::mutate(iter=i))
            }
            readr::write_csv(placebo_res, placebo_res_file)
        }
    }

    # plotting
    y_lim <- y_lims[[outcome_i]]
    y_break <- y_breaks[[outcome_i]]
    g <- ggplot2::ggplot()
    if (placebo) {
        g <- g +
            ggplot2::geom_line(
                data=placebo_res %>% dplyr::filter(iter < 100),
                ggplot2::aes(x=x, y=beta, group=iter), size=0.5, color='grey30', alpha=0.05)
    }
    g <- g +
        ggplot2::scale_y_continuous(name='', breaks=y_break) +
        ggplot2::coord_cartesian(ylim=y_lim, xlim=x_lim, expand=F)
    g <- g +
        ggplot2::geom_line(
            data=main_res, ggplot2::aes(x=x, y=beta), size=1, color=palette[1]) +
        ggplot2::geom_point(
            data=main_res, ggplot2::aes(x=x, y=beta), size=3, color=palette[1]) +
        ggplot2::geom_errorbar(
            data=main_res %>% dplyr::filter(x > 0),
            ggplot2::aes(x=x, y=beta, ymin=beta - 1.96 * se, ymax=beta + 1.96 * se),
            color=palette[1], width=.1) +
        ggplot2::scale_x_continuous(
            name='Cash Infusion (per 0.012 sq km)',
            breaks=c(0, 1, 2, 3),
            labels=c('$0', '$1000', '$2000', '>$2000')) +
        ggplot2::ggtitle(paste0(titles[outcome_i], "\nPoint Estimate: ",
                                round(linear_effect$beta[1], 3), ", 95% CI: [",
                                round(linear_effect$beta[1] - 1.96 * linear_effect$se[1], 3), ", ",
                                round(linear_effect$beta[1] + 1.96 * linear_effect$se[1], 3), "]")) +
        ggplot2::annotate(x=x_lim[1], xend=x_lim[1],
                          y=y_break[1], yend=y_break[length(y_break)], lwd=0.75, geom="segment") +
        g_style
    ggplot2::ggsave(paste0(output_dir, col_ys[outcome_i], '_ate.pdf'), g, width=4, height=2)
}
