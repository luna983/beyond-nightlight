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
    res <- tibble::tibble(beta=se$betahat[1, 1], se=se$conleySE)
    return(res)
}


main <- function(input_file, output_figure_file, output_raw_data_file) {
    col_ys <- c('area_sum', 'tin_area_sum', 'nightlight')
    col_label_ys <- c('building_footprint', 'tin_roof_area', 'night_light')
    titles <- c(expression(paste(bold('a'), '    Building Footprint (', m^2, ')')),
                expression(paste(bold('b'), '    Tin-roof Area (', m^2, ')')),
                expression(paste(bold('c'), '    Night Light (nW·', cm^-2, '·', sr^-1, ')')))
    colors <- c('#DB4437', '#4285F4', '#F4B400')
    y_breaks <- list(
        c(-25, 0, 25, 50), c(-25, 0, 25, 50, 75), c(-0.05, 0, 0.05))
    y_lims <- list(
        c(-30, 55), c(-30, 80), c(-0.055, 0.055))
    x_lim <- c(-0.2, 3.5)

    # load data
    if (file.exists(output_raw_data_file)) {
        cat('Loading cache file:', output_raw_data_file, '\n')
        output_raw_data <- readr::read_csv(
            output_raw_data_file,
            # to suppress warnings
            col_type=readr::cols())
    } else {
        cat('Generating cache file:', output_raw_data_file, '\n')
        df <- readr::read_csv(
            input_file,
            # to suppress warnings
            col_type=readr::cols())
        n_iter <- stringr::str_detect(colnames(df),
                                    'treat_eligible_placebo') %>% sum()
        output_raw_data <- tibble::tibble()
        for (outcome_i in c(1:length(col_ys))) {
            cat('========= Outcome:', col_label_ys[outcome_i], '=========\n')
            # binned effect
            main_res <- regress_bin(df=df, col_y=col_ys[outcome_i])
            main_res %<>% tibble::add_column(
                outcome=col_label_ys[outcome_i], iter=NA, placebo=0)
            output_raw_data %<>% rbind(main_res)
            # linear effect
            linear_effect <- regress_linear(df=df, col_y=col_ys[outcome_i])
            sample_size <- nrow(df %>% tidyr::drop_na(!! col_ys[outcome_i]))
            num_fixed_effects <- length(unique(df$eligible))
            t_stat <- linear_effect$beta[1] / linear_effect$se[1]
            degree_of_freedom <- sample_size - num_fixed_effects - 1
            p_value <- pt(q=abs(t_stat), df=degree_of_freedom, lower.tail=FALSE) * 2
            cat(sprintf(
                paste0(
                    "N = %i;\n",
                    "Point Estimate: %.6f; Standard Error: %.6f;\n",
                    "95%% CI: [%.6f, %.6f];\n",
                    "(two-sided) t-stat: %.6f (df = %i); p-value: %.6f;\n"),
                sample_size,
                linear_effect$beta[1],
                linear_effect$se[1],
                linear_effect$beta[1] - 1.96 * linear_effect$se[1],
                linear_effect$beta[1] + 1.96 * linear_effect$se[1],
                t_stat,
                degree_of_freedom,
                p_value))
            linear_effect %<>% tibble::add_column(
                outcome=col_label_ys[outcome_i], iter=NA, placebo=0, x=NA)
            output_raw_data %<>% rbind(linear_effect)
            # placebo
            cat('Placebo estimation in progress\n')
            for (i in c(0:(n_iter - 1))) {
                cat('.')
                res <- regress_bin(df=df, col_x=sprintf('treat_eligible_placebo%02d', i),
                                col_y=col_ys[outcome_i], compute_se=F)
                res %<>% tibble::add_column(
                    outcome=col_label_ys[outcome_i], iter=i, placebo=1)
                output_raw_data %<>% rbind(res)
            }
            cat('\n')
        }
        readr::write_csv(output_raw_data, output_raw_data_file)
    }
    # plotting
    results <- list()
    for (outcome_i in c(1:length(col_ys))) {
        y_lim <- y_lims[[outcome_i]]
        y_break <- y_breaks[[outcome_i]]
        color <- colors[outcome_i]
        main_res <- output_raw_data %>% dplyr::filter(
            (!is.na(x)) & (outcome == col_label_ys[outcome_i]) & (placebo == 0))
        linear_effect <- output_raw_data %>% dplyr::filter(
            is.na(x) & (outcome == col_label_ys[outcome_i]) & (placebo == 0))
        placebo_res <- output_raw_data %>% dplyr::filter(
            (outcome == col_label_ys[outcome_i]) & (placebo == 1))
        subtitle <- sprintf(
            "      Point Estimate: %.3f, 95%% CI: [%.3f, %.3f]",
            linear_effect$beta[1],
            linear_effect$beta[1] - 1.96 * linear_effect$se[1],
            linear_effect$beta[1] + 1.96 * linear_effect$se[1])
        g <- ggplot2::ggplot()
        g <- g +
            ggplot2::geom_line(
                data=placebo_res,
                ggplot2::aes(x=x, y=beta, group=iter), size=0.5, color='grey30', alpha=0.05)
        g <- g +
            ggplot2::scale_y_continuous(name='', breaks=y_break) +
            ggplot2::coord_cartesian(ylim=y_lim, xlim=x_lim, expand=F)
        g <- g +
            ggplot2::geom_line(
                data=main_res, ggplot2::aes(x=x, y=beta), color=color, size=1) +
            ggplot2::geom_point(
                data=main_res, ggplot2::aes(x=x, y=beta), color=color, size=3) +
            ggplot2::geom_errorbar(
                data=main_res %>% dplyr::filter(x > 0),
                ggplot2::aes(x=x, y=beta, ymin=beta - 1.96 * se, ymax=beta + 1.96 * se),
                color=color, width=.1) +
            ggplot2::scale_x_continuous(
                name='Cash Infusion',
                breaks=c(0, 1, 2, 3),
                labels=c('$0', '$1000', '$2000', '>$2000')) +
            ggplot2::ggtitle(titles[outcome_i], subtitle=subtitle) +
            ggplot2::annotate(x=x_lim[1], xend=x_lim[1],
                            y=y_break[1], yend=y_break[length(y_break)], lwd=0.75, geom="segment") +
            ggplot2::theme_bw() +
            ggplot2::theme(
                panel.border=ggplot2::element_blank(),
                panel.grid.major=ggplot2::element_blank(),
                panel.grid.minor=ggplot2::element_blank(),
                axis.title=ggplot2::element_text(size=11),
                axis.text=ggplot2::element_text(size=11),
                axis.text.x=ggplot2::element_text(vjust=5),
                plot.title=ggplot2::element_text(size=11),
                plot.subtitle=ggplot2::element_text(size=11),
                plot.title.position="plot",
                plot.caption.position="plot",
                axis.ticks.length=ggplot2::unit(.07, "in"),
                axis.ticks.x=ggplot2::element_blank())
        results[[outcome_i]] <- g
    }
    ggplot2::ggsave(output_figure_file,
                    gridExtra::arrangeGrob(results[[1]], results[[2]], results[[3]], nrow=3),
                    width=4, height=7)
}

# Creates the main figure
# main(input_file = 'data/Siaya/Merged/main_res0.0010.csv',
#      output_figure_file = 'output/fig-ate/fig-ate-raw.pdf',
#      output_raw_data_file = 'fig_raw_data/fig-ate.csv')

# Sensitivity checks
# main(input_file = 'data/Siaya/Merged/main_res0.0005.csv',
#      output_figure_file = 'output/fig-ate/fig-ate-res-sensitivity-0.0005-raw.pdf',
#      output_raw_data_file = 'output/fig-ate/archive_data/fig-ate-res-sensitivity-0.0005.csv')
# main(input_file = 'data/Siaya/Merged/main_res0.0015.csv',
#      output_figure_file = 'output/fig-ate/fig-ate-res-sensitivity-0.0015-raw.pdf',
#      output_raw_data_file = 'output/fig-ate/archive_data/fig-ate-res-sensitivity-0.0015.csv')
# main(input_file = 'data/Siaya/Merged/main_res0.0020.csv',
#      output_figure_file = 'output/fig-ate/fig-ate-res-sensitivity-0.0020-raw.pdf',
#      output_raw_data_file = 'output/fig-ate/archive_data/fig-ate-res-sensitivity-0.0020.csv')
main(input_file = 'data/Siaya/Merged/main_res0.0025.csv',
     output_figure_file = 'output/fig-ate/fig-ate-res-sensitivity-0.0025-raw.pdf',
     output_raw_data_file = 'output/fig-ate/archive_data/fig-ate-res-sensitivity-0.0025.csv')
