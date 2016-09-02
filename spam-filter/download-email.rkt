#lang racket
(require net/imap)

(parameterize ([imap-port-number 993])
  (let-values ([(imap total recent)
                (imap-connect "imap.gmail.com" "skanaley" "blackwid0w" "INBOX" #:tls? #t)])
    (let-values ([(t r) (imap-examine imap "INBOX")])
      (printf "~a ~a~n" t r))
      (displayln (imap-get-expunges imap))
      (with-output-to-file "nonspams" #:exists 'truncate
        (λ ()
          (write (imap-get-messages imap (range 6000 6038) '(body)))))
    (imap-disconnect imap)))