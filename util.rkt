#lang racket/base
(require racket/runtime-path
         racket/file
         racket/string
         racket/list
         racket/function)

(provide v.
         transpose
         load-linear-data
         load-housing-data
         load-multinomial-data)

(define-runtime-path linear-data "linear.txt")
(define-runtime-path housing-data "data/housing.data")
(define-runtime-path multinomial-data "multinomial.txt")

(define (v. as bs)
  (for/sum ([a as]
            [b bs])
    (* a b)))

(define (transpose xss)
  (apply map list xss))

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