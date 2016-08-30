#lang racket/base
(require racket/function
         racket/math
         racket/list
         plot
         "util.rkt")

(define (linear-regression/gaussian/gradient-descent
         #:good-cost-diff good-cost-diff
         #:learning-rate learning-rate
         #:debug? [debug? #f]
         xss ys)
  (let ([m (length ys)]
        [xsst (transpose xss)])
    (let loop ([iter 0]
               [cost +inf.0]
               [cs (make-list (length (car xss)) 0)])
      (let* ([h (curry v. cs)]
             [diffs (for/list ([xs xss]
                               [y ys])
                      (- y (h xs)))]
             [sum-sqr (for/sum ([d diffs])
                        (sqr d))]
             [cost* (* 1/2 sum-sqr)])
        (when (= 0 (modulo iter 500))
          (printf "J=~a~nRMS=~a~nTHETA=~a~n~n"
                  cost*
                  (sqrt (/ sum-sqr m))
                  cs))
        (if (<= (abs (- cost* cost))
                good-cost-diff)
            h
            (loop (add1 iter)
                  cost*
                  (let ([rand-i (random (length cs))])
                    (for/list ([xst xsst]
                               [(c i) (in-indexed cs)])
                      (+ c (* learning-rate
                              (let ([derived-deriv (for/sum ([d diffs]
                                                             [x xst])
                                                     (* x d))])
                                (when (and debug? (= i rand-i) (= 0 (modulo iter 50)))
                                  (let* ([e 0.0001]
                                         [J (λ (f)
                                              (let-values ([(l r) (split-at cs i)])
                                                (let ([cs (append l (list (f (car r) e)) (cdr r))])
                                                  (* -1/2 (for/sum ([xs xss]
                                                                    [y ys])
                                                            (sqr (- y (v. xs cs))))))))]
                                         [approx-deriv (/ (- (J +) (J -))
                                                          (* 2 e))])
                                    (printf "Gradients:~nderived: ~a~napprox: ~a~n~n" derived-deriv approx-deriv)))
                                derived-deriv)))))))))))

(module+ main
  (let-values ([(xss ys) (load-housing-data)])
    (let ([h (time (linear-regression/gaussian/gradient-descent
                    #:good-cost-diff 0.01
                    #:learning-rate 0.00000001
                    #:debug? #t
                    xss ys))])
      (plot #:x-min 0
            #:x-max 510
            #:y-min -10
            #:y-max 50
            (list (points #:sym 'dot
                          (for/list ([(y i) (in-indexed ys)])
                            (list i y)))
                  (points #:sym 'plus
                          #:color 'blue
                          (for/list ([(xs i) (in-indexed xss)])
                            (list i (h xs)))))))))