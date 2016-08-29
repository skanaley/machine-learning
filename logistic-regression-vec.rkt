#lang typed/racket/base
(require math/matrix
         plot/typed)

(require/typed "util.rkt"
               [v. (-> (Listof Real) (Listof Real) Real)]
               [sigmoid (-> Real Nonnegative-Real)]
               [load-mnist-images (-> (Listof (Listof Real)))]
               [load-mnist-labels (-> (Listof Real))])

(: logistic-regression/bernoulli/gradient-descent (->* (#:good-cost-diff Real
                                                                         #:learning-rate Real
                                                                         (Listof (Listof Real))
                                                                         (Listof Real))
                                                       (-> (Listof Real) Real)))
(define (logistic-regression/bernoulli/gradient-descent
         #:good-cost-diff good-cost-diff
         #:learning-rate learning-rate
         xss ys)
  (let* ([X (list*->matrix xss)]
         [Xt (matrix-transpose X)]
         [Y (->col-matrix ys)]
         [m (matrix-num-rows X)]
         [n (matrix-num-cols X)])
    (let loop ([iter 0]
               [cost : Number +inf.0]
               [C (build-matrix n 1 (λ _ (* 0.001 (random))))])
      (let* ([H (matrix-map sigmoid (matrix* X C))]
             [D (matrix- Y H)]
             [sum-sqr (matrix-dot D)]
             [cost* : Number (- (+ (matrix-dot Y (matrix-map log H))
                                   (matrix-dot (matrix-map (λ (y) (- 1 y)) Y)
                                               (matrix-map (λ (h) (log (- 1 h))) H))))])
        (when #t;(= 0 (modulo iter 10)) too slow anyway
          (printf "iter=~a~nJ=~a~nRMS=~a~n~n";THETA=~a~n~n" too many coeffs
                  iter
                  cost*
                  (sqrt (/ sum-sqr m))
                  ;cs))
                  ))
        (if (= 20 iter)#|(<= (abs (- cost* cost))
                good-cost-diff)|#
            (λ (xs) (sigmoid (v. xs (matrix->list C))))
            (loop (add1 iter)
                  cost*
                  (matrix+ C (matrix-scale (matrix* Xt D) learning-rate))))))))

(module+ main
  (let ([xss (load-mnist-images)]
        [ys (load-mnist-labels)])
    (let-values ([(xss ys) (for/fold ([xss : (Listof (Listof Real)) '()]
                                      [ys : (Listof Real) '()])
                                     ([xs xss]
                                      [y ys]
                                      #:when (< y 2))
                             (values (cons xs xss) (cons y ys)))])
      (let ([h (time (logistic-regression/bernoulli/gradient-descent
                      #:good-cost-diff 1
                      #:learning-rate 0.0005
                      xss ys))])
        (let-values ([(r w) (for/fold ([right : Nonnegative-Integer 0] [wrong : Nonnegative-Integer 0])
                                      ([xs xss]
                                       [y ys])
                              (if (= y (round (h xs)))
                                  (values (add1 right) wrong)
                                  (values right (add1 wrong))))])
          (printf "accuracy: ~a~n" (exact->inexact (/ r (+ r w)))))))))