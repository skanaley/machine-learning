#lang racket/base
(require racket/runtime-path
         racket/string
         racket/list
         racket/function)

(define-runtime-path nonspam-path "nonspams")
(define-runtime-path spam-path "spams")

(define (email->words e)
  (for/list ([w (string-split (bytes->string/utf-8 (car e)))]
             #:when (regexp-match? #px"^[[:alpha:]]+(!\\?\\.)?$" w))
    (string-downcase w)))

(define nonspams (map email->words (with-input-from-file nonspam-path read)))
(define spams (map email->words (with-input-from-file spam-path read)))

(define all-words
  (for*/fold ([h (hash)])
             ([email (append spams nonspams)]
              [word email])
    (hash-update h word add1 0)))

(define dictionary
  (for/hash ([(k v) all-words]
             #:when (>= v 1))
    (values k v)))

(define phi-xs0
  (for/hash ([(k _) dictionary])
    (values k
            (/ (add1 (for/sum ([e nonspams])
                       (count (curry string=? k) e)))
               (+ (hash-count dictionary)
                  (for/sum ([e nonspams])
                    (length e)))))))

(define phi-xs1
  (for/hash ([(k _) dictionary])
    (values k
            (/ (add1 (for/sum ([e spams])
                       (count (curry string=? k) e)))
               (+ (hash-count dictionary)
                  (for/sum ([e spams])
                    (length e)))))))

(define phi-y
  (/ (add1 (length spams))
     (+ 2 (length spams) (length nonspams))))

(define (spam? xs)
  (> (* phi-y
        (for/product ([x xs])
          (hash-ref phi-xs1 x (/ 1 phi-y))))
     (* (- 1 phi-y)
        (for/product ([x xs])
          (hash-ref phi-xs0 x (/ 1 (- 1 phi-y)))))))

(define spammiest-words
  (sort (hash->list (for/hash ([(k v) phi-xs1])
                      (values k (/ v (hash-ref phi-xs0 k 0.0)))))
        (λ (a b) (> (cdr a) (cdr b)))))