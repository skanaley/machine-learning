#lang typed/racket/base
(require math/matrix
         plot/typed)

(require/typed "util.rkt"
               [v. (-> (Listof Real) (Listof Real) Real)]
               [load-housing-data (-> (values (Listof (Listof Real))
                                              (Listof Real)))])

(: linear-regression/gaussian/gradient-descent (->* (#:good-cost-diff Real
                                                                      #:learning-rate Real
                                                                      (Listof (Listof Real))
                                                                      (Listof Real))
                                                    (-> (Listof Real) Real)))
(define (linear-regression/gaussian/gradient-descent
         #:good-cost-diff good-cost-diff
         #:learning-rate learning-rate
         xss ys)
  (let* ([X (list*->matrix xss)]
         [Y (->col-matrix ys)]
         [Xt (matrix-transpose X)]
         [m (matrix-num-rows X)]
         [n (matrix-num-cols X)])
    (let loop ([iter 0]
               [cost : Real +inf.0]
               [C : (Matrix Real) (make-matrix n 1 0)])
      (let* ([H (matrix* X C)]
             [D (matrix- Y H)]
             [sum-sqr (matrix-dot D)]
             [cost* (* 1/2 sum-sqr)])
        (when (= 0 (modulo iter 500))
          (printf "J=~a~nRMS=~a~nTHETA=~a~n~n"
                  cost*
                  (sqrt (/ sum-sqr m))
                  C))
        (if (<= (abs (- cost* cost))
                good-cost-diff)
            (λ (xs) (v. xs (matrix->list C)))
            (loop (add1 iter)
                  cost*
                  (matrix+ C (matrix-scale (matrix* Xt D) learning-rate))))))))

(module+ main
  (let-values ([(xss ys) (load-housing-data)])
    (let ([h (time (linear-regression/gaussian/gradient-descent
                    #:good-cost-diff 0.01
                    #:learning-rate 0.00000001
                    xss ys))])
      (plot #:x-min 0
            #:x-max 510
            #:y-min -10
            #:y-max 50
            (list (points #:sym 'dot
                          (for/list: : (Listof (List Real Real))
                            ([(y i) (in-indexed ys)])
                            (list i y)))
                  (points #:sym 'plus
                          #:color 'blue
                          (for/list: : (Listof (List Real Real))
                            ([(xs i) (in-indexed xss)])
                            (list i (h xs)))))))))