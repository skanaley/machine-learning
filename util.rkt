#lang racket/base
(require racket/runtime-path
         racket/file
         racket/string
         racket/list
         racket/function
         racket/match
         racket/math)

(provide transpose
         ngroup
         v.
         sigmoid
         load-linear-data load-housing-data
         load-logistic-data load-mnist-images load-mnist-labels load-test-mnist-images load-test-mnist-labels
         load-multinomial-data)

(define-runtime-path linear-data "linear.txt")
(define-runtime-path housing-data "data/housing.data")
(define-runtime-path logistic-data "logistic.txt")
(define-runtime-path mnist-images "data/train-images-idx3-ubyte")
(define-runtime-path mnist-labels "data/train-labels-idx1-ubyte")
(define-runtime-path test-mnist-images "data/t10k-images-idx3-ubyte")
(define-runtime-path test-mnist-labels "data/t10k-labels-idx1-ubyte")
(define-runtime-path multinomial-data "multinomial.txt")

(define (transpose xss)
  (apply map list xss))

(define (split-at* n xs)
  (let loop ([xs xs] [n n] [l '()] [r '()])
    (match xs
      ['() (values (reverse l) (reverse r))]
      [(cons x xs*)
       (if (= 0 n)
           (values (reverse l) (reverse (append (reverse xs) r)))
           (loop xs* (sub1 n) (cons x l) r))])))

(define (ngroup n xs)
  (let loop ([xs xs] [out '()])
    (if (null? xs)
        (reverse out)
        (let-values ([(l r) (split-at* n xs)])
          (loop r (cons l out))))))

(define (v. as bs)
  (for/sum ([a as]
            [b bs])
    (* a b)))

(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(define (generate-linear-data)
  (with-output-to-file linear-data #:exists 'truncate
    (λ ()
      (write (for/list ([x 100])
               (list 1 x)))
      (write (for/list ([x 100])
               (+ (* 3 x) -50 (random 101)))))))

(define (load-linear-data)
  (with-input-from-file linear-data
    (λ ()
      (values (read) (read)))))

(define (load-housing-data)
  (let ([rows (transpose (sort (for/list ([l (file->lines housing-data)])
                                 (map string->number (string-split l)))
                               (λ (a b) (< (last a) (last b)))))])
    (let-values ([(xss ys) (split-at rows (sub1 (length rows)))])
      (values (map (curry cons 1) (transpose xss))
              (car ys)))))

(define (generate-logistic-data)
  (with-output-to-file logistic-data #:exists 'truncate
    (λ ()
      (write (for/list ([x 100])
               (list 1 x)))
      (write (for/list ([x 100])
               ((if (= 0 (random 15))
                    (curry - 1)
                    identity)
                (if (>= x 50) 1 0)))))))

(define (load-logistic-data)
  (with-input-from-file logistic-data
    (λ ()
      (values (read) (read)))))

(define-values (load-mnist-images load-mnist-labels load-test-mnist-images load-test-mnist-labels)
  (let ([r4 (λ (n err)
              (unless (= n (integer-bytes->integer (read-bytes 4) #t #t))
                (error 'load-mnist-data err)))])
    (values (λ ()
              (with-input-from-file mnist-images
                (λ ()
                  (r4 2051 "wrong image file magic number")
                  (r4 60000 "wrong number of images")
                  (r4 28 "wrong number of rows")
                  (r4 28 "wrong number of columns")
                  (let* ([i-size (* 28 28)]
                         [t-size (* i-size 60000)]
                         [bss (for/list ([i 60000])
                                (read-bytes i-size))])
                    (displayln 'bss)
                    (let ([m (/ (for*/sum ([bs bss]
                                           [b bs])
                                  b)
                                t-size)])
                      (displayln m)
                      (let ([s (sqrt (/ (for*/sum ([bs bss]
                                                   [b bs])
                                          (sqr (- b m)))
                                        t-size))])
                        (displayln s)
                        (for/list ([bs bss])
                          (cons 1 (for/list ([b bs])
                                    (/ (- b m) s))))))))))
            (λ ()
              (with-input-from-file mnist-labels
                (λ ()
                  (r4 2049 "wrong label file magic number")
                  (r4 60000 "wrong number of labels")
                  (bytes->list (read-bytes 60000)))))
            (λ ()
              (with-input-from-file test-mnist-images
                (λ ()
                  (r4 2051 "wrong image file magic number")
                  (r4 10000 "wrong number of images")
                  (r4 28 "wrong number of rows")
                  (r4 28 "wrong number of columns")
                  (let* ([i-size (* 28 28)]
                         [t-size (* i-size 10000)]
                         [bss (for/list ([i 10000])
                                (read-bytes i-size))])
                    (displayln 'bss)
                    (let ([m (/ (for*/sum ([bs bss]
                                           [b bs])
                                  b)
                                t-size)])
                      (displayln m)
                      (let ([s (sqrt (/ (for*/sum ([bs bss]
                                                   [b bs])
                                          (sqr (- b m)))
                                        t-size))])
                        (displayln s)
                        (for/list ([bs bss])
                          (cons 1 (for/list ([b bs])
                                    (/ (- b m) s))))))))))
            (λ ()
              (with-input-from-file test-mnist-labels
                (λ ()
                  (r4 2049 "wrong label file magic number")
                  (r4 10000 "wrong number of labels")
                  (bytes->list (read-bytes 10000))))))))

(define (generate-multinomial-data)
  (with-output-to-file multinomial-data #:exists 'truncate
    (λ ()
      (write (for/list ([x 100])
               (list 1 x)))
      (write (for/list ([x 100])
               (cond [(< x 33) 0]
                     [(< x 67) 2]
                     [else 1]))))))

(define (load-multinomial-data)
  (with-input-from-file multinomial-data
    (λ ()
      (values (read) (read)))))